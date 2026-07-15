#!/usr/bin/env python3
"""In-RAM read/write throughput benchmark -> two Mev/s matrices.

What & why
----------
Real event recordings are far too large to decode in one shot, so production
code *always* reads them in chunks. This benchmark therefore measures only the
**chunked** read path (never ``read_all``) and reports **throughput in
M events/s**, which -- unlike wall-clock time -- stays comparable across files of
any size, format, and library.

Everything runs on a **tmpfs (RAM disk, default /dev/shm)** so disk I/O never
enters the measurement; after warmup a tmpfs file is served from the page cache
at RAM speed, so only decode/encode cost is timed. The same is true for every
library, so the comparison stays fair.

Output is two matrices (rows = format, cols = library/variant, cells = Mev/s):

* **Read** matrix, with the evutils read path split into the variants whose cost
  we want to see separately:

  - ``evutils n_events`` -- ``EventReader`` slicing by a fixed event count
    (the common chunked-read path).
  - ``evutils delta_t``  -- ``EventReader`` slicing by a fixed time window;
    the gap to ``n_events`` on the same row *is* the delta_t slicing penalty.
  - ``evutils async``    -- ``EventReader(async_read=True)`` (background-thread
    prefetch; n_events slicing under the hood).
  - ``evutils stream``   -- ``EventStreamer``: raw chunked decode, no slicing.
    The throughput ceiling and the true peer of expelliarmus' ``read_chunk``.

  plus any installed comparison library (expelliarmus, evlib, evt3, openeb) --
  each auto-skips (its column disappears) if not importable.

* **Write** matrix: ``evutils`` vs any installed comparison writer
  (expelliarmus).

Data & memory
-------------
Real recordings ship only as evt2/evt3/evt21. The benchmark decodes the first
``--events N`` events of one real recording into a single in-RAM array (the
decoded array is the thing that can't fit whole, so it is capped), then, one
format at a time, writes that identical event set to the RAM disk and benchmarks
reads + writes on it before deleting it. Using one event set for every format
makes the matrices directly comparable.

Usage
-----
    python benchmarks/throughput.py                       # download 'normal', defaults
    python benchmarks/throughput.py --dataset small --events 2_000_000 --repeats 2
    python benchmarks/throughput.py --file /path/to/rec.raw
    python benchmarks/throughput.py --libs evutils,expelliarmus,evlib,evt3,openeb
    python benchmarks/throughput.py --markdown docs.md --json bench.json
"""

from __future__ import annotations

import argparse
import gc
import os
import sys
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np

from evutils.io import EventReader, EventStreamer, EventWriter

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(_REPO_ROOT / "tests"))  # shared download/discovery helpers

# On-disk extension per format. EVT variants share the .raw container (the
# variant lives in the header and is selected via ``format=``).
_EXT = {
    "evt3": ".raw", "evt2": ".raw", "evt21": ".raw", "evt4": ".raw",
    "dat": ".dat", "csv": ".csv",
    "npz": ".npz", "hdf5": ".h5", "aer": ".aer",
}
_EVT_FMTS = {"evt3", "evt2", "evt21", "evt4"}
_CORE_FORMATS = ["evt3", "evt2", "evt21", "evt4", "dat", "csv"]
_CONTAINER_FORMATS = ["npz", "hdf5", "aer"]

#: Libraries enabled by default. Matched as name prefixes so the single token
#: "evutils" enables all four evutils variants. evt3 + openeb are opt-in.
DEFAULT_LIBS = "evutils,expelliarmus,evlib"

_NA = "-"  # rendered in a cell a library cannot handle


def _enabled(name: str, tokens: list[str]) -> bool:
    """A library is enabled if any --libs token equals or prefixes its name, so
    ``evutils`` selects every ``evutils ...`` variant at once."""
    return any(name == t or name.startswith(t) for t in tokens)


# --------------------------------------------------------------------------- #
# Read adapters: (path, fmt, chunk, delta_t) -> event count. Chunked only.
# Each imports its library lazily so a missing lib skips instead of erroring.
# --------------------------------------------------------------------------- #
def _read_evutils_n_events(path: str, fmt: str, chunk: int, delta_t: int) -> tuple[int, int]:
    total = 0
    n_pos = 0
    with EventReader(path, n_events=chunk) as r:
        for c in r:
            total += len(c)
            n_pos += np.count_nonzero(c.p == 1)
    return total, n_pos


def _read_evutils_delta_t(path: str, fmt: str, chunk: int, delta_t: int) -> tuple[int, int]:
    total = 0
    n_pos = 0
    with EventReader(path, delta_t=delta_t) as r:
        for c in r:
            total += len(c)
            n_pos += np.count_nonzero(c.p == 1)
    return total, n_pos


def _read_evutils_async(path: str, fmt: str, chunk: int, delta_t: int) -> tuple[int, int]:
    total = 0
    n_pos = 0
    with EventReader(path, n_events=chunk, async_read=True) as r:
        for c in r:
            total += len(c)
            n_pos += np.count_nonzero(c.p == 1)
    return total, n_pos


def _read_evutils_stream(path: str, fmt: str, chunk: int, delta_t: int) -> tuple[int, int]:
    # EventStreamer is the low-level chunked decoder -- raw decode, no windowing;
    # chunks alias a reused buffer, so a streaming consumer processes each in
    # place (here: count) and moves on. The true peer of read_chunk.
    total = 0
    n_pos = 0
    for c in EventStreamer(path):
        total += len(c)
        n_pos += np.count_nonzero(c.p == 1)
    return total, n_pos


def _read_expelliarmus(path: str, fmt: str, chunk: int, delta_t: int) -> tuple[int, int]:
    from expelliarmus import Wizard  # type: ignore

    wiz = Wizard(encoding=fmt, chunk_size=chunk)
    wiz.set_file(str(path))
    n_pos = 0
    total = 0
    for c in wiz.read_chunk():
        n_pos += np.count_nonzero(c["p"] == 1)
        total += len(c)
    return total, n_pos

def _read_expelliarmus_delta_t(path: str, fmt: str, chunk: int, delta_t: int) -> tuple[int, int]:
    from expelliarmus import Wizard  # type: ignore

    wiz = Wizard(encoding=fmt, chunk_size=chunk)
    wiz.set_file(str(path))
    wiz.set_time_window(delta_t)
    n_pos = 0
    total = 0
    for c in wiz.read_time_window():
        n_pos += np.count_nonzero(c["p"] == 1)
        total += len(c) 
    return total, n_pos


def _read_evlib(path: str, fmt: str, chunk: int, delta_t: int) -> tuple[int, int]:
    # evlib has no fixed-size chunk API; it loads lazily and collects in
    # streaming mode (bounded memory), the closest comparable to a chunked read.
    import evlib  # type: ignore

    return int(evlib.load_events(str(path)).collect(engine="streaming").height), 0


def _read_evt3(path: str, fmt: str, chunk: int, delta_t: int) -> tuple[int, int]:
    import evt3  # type: ignore

    return int(len(evt3.decode_file(str(path)))), 0


def _read_openeb(path: str, fmt: str, chunk: int, delta_t: int) -> tuple[int, int]:
    from metavision_core.event_io import RawReader  # type: ignore

    reader = RawReader(str(path))
    total = 0
    while not reader.is_done():
        total += len(reader.load_n_events(chunk))
    return total, 0


#: (column label, formats it can read, reader). evutils variants first; column
#: order in the matrix follows this list.
READ_ADAPTERS: list[tuple[str, tuple[str, ...], Callable[[str, str, int, int], tuple[int, int]]]] = [
    ("evutils n_events", ("evt3", "evt2", "evt21", "evt4", "dat", "csv", "npz", "hdf5", "aer"), _read_evutils_n_events),
    ("evutils delta_t",  ("evt3", "evt2", "evt21", "evt4", "dat", "csv", "npz", "hdf5", "aer"), _read_evutils_delta_t),
    ("evutils async",    ("evt3", "evt2", "evt21", "evt4", "dat", "csv", "npz", "hdf5", "aer"), _read_evutils_async),
    ("evutils stream",   ("evt3", "evt2", "evt21", "evt4", "dat", "csv", "npz", "hdf5", "aer"), _read_evutils_stream),
    ("expelliarmus",     ("evt3", "evt2", "dat"), _read_expelliarmus),
    ("expelliarmus delta_t", ("evt3", "evt2", "dat"), _read_expelliarmus_delta_t),
    ("evlib",            ("evt3", "evt2"), _read_evlib),
    ("evt3",             ("evt3",), _read_evt3),
    ("openeb",           ("evt3", "evt2", "evt21"), _read_openeb),
]


# --------------------------------------------------------------------------- #
# Write adapters: (path, fmt, data) -> event count written.
# --------------------------------------------------------------------------- #
def _write_evutils(path: str, fmt: str, data: np.ndarray) -> int:
    if fmt in _EVT_FMTS:
        with EventWriter(path, format=fmt) as w:
            w.write(data)
    else:  # dat / csv / npz / hdf5 / aer -- encoder chosen by extension
        with EventWriter(path) as w:
            w.write(data)
    return len(data)


# expelliarmus' structured dtype (aligned, itemsize 16), as returned by
# Wizard.read(). Its save() rejects other layouts, so recast into it.
_EXP_DTYPE = np.dtype([("t", "<i8"), ("x", "<i2"), ("y", "<i2"), ("p", "u1")], align=True)


def _expelliarmus_array(data: np.ndarray) -> np.ndarray:
    out = np.empty(len(data), dtype=_EXP_DTYPE)
    for f in ("t", "x", "y", "p"):
        out[f] = data[f]
    return out


def _write_expelliarmus(path: str, fmt: str, data: np.ndarray) -> tuple[int, int]:
    from expelliarmus import Wizard  # type: ignore

    wiz = Wizard(encoding=fmt)
    wiz.save(str(path), _expelliarmus_array(data))
    return len(data), 0


WRITE_ADAPTERS: list[tuple[str, tuple[str, ...], Callable[[str, str, np.ndarray], tuple[int, int]]]] = [
    ("evutils",      ("evt3", "evt2", "evt21", "evt4", "dat", "csv", "npz", "hdf5", "aer"), _write_evutils),
    ("expelliarmus", ("evt3", "evt2", "dat"), _write_expelliarmus),
]


# --------------------------------------------------------------------------- #
# Data source
# --------------------------------------------------------------------------- #
def _fetch_dataset(size: str, cache_root: Path) -> dict[str, list[Any]]:
    """Download+extract the reference tarball for ``size`` (cached), then parse.

    Reuses the exact helpers + cache layout the test suite uses, so the download
    is shared: ``<repo>/.pytest_cache/d/<subdir>``.
    """
    from conftest_utils import (  # type: ignore
        DATASETS, download_and_extract_github, load_event_files,
    )
    url, tar_name, subdir = DATASETS[size]
    d = cache_root / subdir
    d.mkdir(parents=True, exist_ok=True)
    download_and_extract_github(url, d, tar_name)
    return load_event_files(d)


def _resolve_source(args: argparse.Namespace) -> Path:
    """Return the one real recording to decode the payload from."""
    if args.file:
        p = Path(args.file)
        if not p.is_file():
            raise SystemExit(f"--file {p} not found")
        return p

    override = os.environ.get("EVUTILS_BENCH_DATA")
    if override:
        from conftest_utils import load_event_files  # type: ignore

        d = Path(override)
        if not d.is_dir():
            raise SystemExit(f"EVUTILS_BENCH_DATA={override} is not a directory")
        files = load_event_files(d)
    else:
        cache_root = Path(args.cache_dir) if args.cache_dir else _REPO_ROOT / ".pytest_cache" / "d"
        files = _fetch_dataset(args.dataset, cache_root)

    # Prefer an evt3 'hand' recording; fall back to any evt3, then anything.
    for fmt in ("evt3", "evt2", "evt21"):
        lst = files.get(fmt) or []
        hand = next((f for f in lst if "hand" in Path(f.path).name), None)
        if hand:
            return Path(hand.path)
        if lst:
            return Path(lst[0].path)
    for lst in files.values():
        if lst:
            return Path(lst[0].path)
    raise SystemExit("no recordings found in the dataset")


def _load_events_capped(path: Path, cap: int) -> np.ndarray:
    """Load up to ``cap`` events from a real file into an AoS array, streamed so
    a huge file is never fully materialised."""
    parts = []
    total = 0
    for c in EventStreamer(str(path)):
        take = c[: cap - total] if total + len(c) > cap else c
        parts.append(take.to_aos() if hasattr(take, "to_aos") else np.asarray(take).copy())
        total += len(take)
        if total >= cap:
            break
    return np.concatenate(parts) if parts else np.empty(0)


# --------------------------------------------------------------------------- #
# Timing
# --------------------------------------------------------------------------- #
def _time(fn: Callable[[], tuple[int, int]], repeats: int, warmup: int) -> tuple[float, float, int]:
    """Run fn (warmup then repeats). Returns (min_s, mean_s, last_count)."""
    n = 0
    for _ in range(warmup):
        n = fn()
        gc.collect()
    ts = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        n, _ = fn()
        ts.append(time.perf_counter() - t0)
        gc.collect()
    return min(ts), sum(ts) / len(ts), n


def _mev_s(n: int, seconds: float) -> float:
    return n / seconds / 1e6 if seconds > 0 else float("nan")


# --------------------------------------------------------------------------- #
# Benchmark
# --------------------------------------------------------------------------- #
def _run_cell(
    fn: Callable[[], tuple[int, int]], expected: int, repeats: int, warmup: int,
) -> dict[str, Any] | None:
    """Time one (format, library) cell. Returns a result dict, or None on skip."""
    try:
        mn, mean, n = _time(fn, repeats, warmup)
    except Exception as exc:  # missing lib / unsupported build
        print(f"    skipped ({type(exc).__name__}: {exc})")
        return None
    flag = "" if n == expected else f"  !! count={n:,} != {expected:,}"
    return dict(n=n, peak=_mev_s(n, mn), mean=_mev_s(n, mean), mismatch=bool(flag), flag=flag)


def benchmark(
    payload: np.ndarray, formats: list[str], tmpdir: Path, tokens: list[str],
    repeats: int, warmup: int, chunk: int, delta_t: int, do_read: bool, do_write: bool,
) -> dict[str, dict[str, dict[str, Any]]]:
    """Returns results[phase][fmt][lib] = cell dict."""
    n_events = len(payload)
    results: dict[str, dict[str, dict[str, Any]]] = {"read": {}, "write": {}}

    for fmt in formats:
        ext = _EXT.get(fmt, ".raw")
        src = tmpdir / f"bench_{fmt}{ext}"
        # Prepare: write the payload once (untimed) as the read source; also a
        # writer sanity check.
        try:
            _write_evutils(str(src), fmt, payload)
        except Exception as exc:
            print(f"\n[{fmt}] prepare FAILED ({type(exc).__name__}: {exc}); skipping format")
            src.unlink(missing_ok=True)
            continue
        size = src.stat().st_size
        print(f"\n[{fmt}] {size/1e6:.0f} MB  ({size/n_events:.2f} B/ev)  {n_events/1e6:.1f} M events")

        if do_read:
            results["read"][fmt] = {}
            for name, fmts, fn in READ_ADAPTERS:
                if fmt not in fmts or not _enabled(name, tokens):
                    continue
                print(f"  read  {name:18s} ", end="")
                cell = _run_cell(lambda fn=fn: fn(str(src), fmt, chunk, delta_t),
                                 n_events, repeats, warmup)
                if cell:
                    results["read"][fmt][name] = cell
                    print(f"peak {cell['peak']:7.1f}  mean {cell['mean']:7.1f} Mev/s{cell['flag']}")

        if do_write:
            results["write"][fmt] = {}
            for name, fmts, fn in WRITE_ADAPTERS:
                if fmt not in fmts or not _enabled(name, tokens):
                    continue
                out = tmpdir / f"w_{name.split()[0]}_{fmt}{ext}"
                print(f"  write {name:18s} ", end="")
                cell = _run_cell(lambda fn=fn, out=out: fn(str(out), fmt, payload),
                                 n_events, repeats, warmup)
                if cell:
                    results["write"][fmt][name] = cell
                    print(f"peak {cell['peak']:7.1f}  mean {cell['mean']:7.1f} Mev/s")
                out.unlink(missing_ok=True)

        src.unlink(missing_ok=True)
        gc.collect()
    return results


# --------------------------------------------------------------------------- #
# Rendering
# --------------------------------------------------------------------------- #
def _columns(adapters: list[tuple], phase: dict[str, dict[str, Any]]) -> list[str]:
    """Adapter-order column labels that produced at least one result."""
    seen = {lib for row in phase.values() for lib in row}
    return [name for name, *_ in adapters if name in seen]


def _matrix_lines(
    title: str, formats: list[str], cols: list[str],
    phase: dict[str, dict[str, Any]], metric: str, markdown: bool,
) -> list[str]:
    if not cols:
        return [f"{title}: (no results)"]

    def cell(fmt: str, lib: str) -> str:
        c = phase.get(fmt, {}).get(lib)
        if not c:
            return _NA
        v = f"{c[metric]:.1f}"
        return v + " !!" if c["mismatch"] else v

    rows = [f for f in formats if f in phase and phase[f]]
    if markdown:
        out = [f"### {title} (M events/s, {metric}; higher is better)", ""]
        out.append("| format | " + " | ".join(cols) + " |")
        out.append("|" + "---|" * (len(cols) + 1))
        for f in rows:
            out.append(f"| {f} | " + " | ".join(cell(f, c) for c in cols) + " |")
        out.append("")
        return out

    w = max(12, *(len(c) for c in cols))
    out = [f"===== {title} (M events/s, {metric}; higher is better) ====="]
    out.append(f"{'format':8s}" + "".join(f"{c:>{w+2}s}" for c in cols))
    for f in rows:
        out.append(f"{f:8s}" + "".join(f"{cell(f, c):>{w+2}s}" for c in cols))
    return out


def render(results: dict, formats: list[str], metric: str, markdown: bool) -> str:
    blocks = []
    for phase, title, adapters in (
        ("read", "Read throughput", READ_ADAPTERS),
        ("write", "Write throughput", WRITE_ADAPTERS),
    ):
        ph = results.get(phase) or {}
        if not any(ph.values()):
            continue
        cols = _columns(adapters, ph)
        blocks.append("\n".join(_matrix_lines(title, formats, cols, ph, metric, markdown)))
    return ("\n\n" if markdown else "\n\n").join(blocks)


def _dump_json(results: dict, path: Path, meta: dict) -> None:
    import json

    path.write_text(json.dumps({"meta": meta, "results": results}, indent=2))


# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--dataset", default="normal", choices=["small", "normal", "large"],
                    help="reference-data tier to download (default normal)")
    ap.add_argument("--file", default=None,
                    help="decode the payload from THIS real recording instead of "
                         "a downloaded dataset")
    ap.add_argument("--cache-dir", default=None,
                    help="download cache root (default <repo>/.pytest_cache/d, "
                         "shared with the test suite)")
    ap.add_argument("--events", type=int, default=20_000_000,
                    help="events to decode into the in-RAM payload (default 20M)")
    ap.add_argument("--formats", default=",".join(_CORE_FORMATS),
                    help=f"comma list (default {','.join(_CORE_FORMATS)})")
    ap.add_argument("--containers", action="store_true",
                    help=f"also benchmark container formats ({','.join(_CONTAINER_FORMATS)})")
    ap.add_argument("--libs", default=DEFAULT_LIBS,
                    help=f"comma list of libraries to enable, prefix match "
                         f"(default {DEFAULT_LIBS!r}; add evt3,openeb to include them)")
    ap.add_argument("--repeats", type=int, default=5, help="timed repeats per cell")
    ap.add_argument("--warmup", type=int, default=1, help="warmup runs per cell")
    ap.add_argument("--chunk", type=int, default=1_000_000,
                    help="read chunk size in events (n_events / async / openeb)")
    ap.add_argument("--delta-t", type=int, default=10_000,
                    help="time window (us) for the delta_t read variant")
    ap.add_argument("--tmpfs", default="/dev/shm", help="RAM-disk dir (default /dev/shm)")
    ap.add_argument("--metric", default="peak", choices=["peak", "mean"],
                    help="metric shown in the matrices (default peak)")
    ap.add_argument("--markdown", default=None, help="write the matrices as markdown to this path")
    ap.add_argument("--json", default=None, help="dump full stats as JSON to this path")
    ap.add_argument("--no-read", action="store_true")
    ap.add_argument("--no-write", action="store_true")
    args = ap.parse_args()

    tmpdir = Path(args.tmpfs)
    if not (tmpdir.is_dir() and os.access(tmpdir, os.W_OK)):
        raise SystemExit(f"tmpfs dir {tmpdir} not writable")
    tokens = [t.strip() for t in args.libs.split(",") if t.strip()]
    formats = [f.strip() for f in args.formats.split(",") if f.strip()]
    if args.containers:
        formats += [f for f in _CONTAINER_FORMATS if f not in formats]

    source = _resolve_source(args)
    print(f"Source recording: {source}")
    print(f"Decoding up to {args.events/1e6:.0f} M events into the in-RAM payload ...")
    payload = _load_events_capped(source, args.events)
    if len(payload) == 0:
        raise SystemExit("decoded 0 events from the source recording")
    print(f"Payload: {len(payload):,} events  (~{payload.nbytes/1e9:.2f} GB in RAM)")
    print(f"Formats: {', '.join(formats)}   Libraries: {', '.join(tokens)}")

    results = benchmark(
        payload, formats, tmpdir, tokens, args.repeats, args.warmup,
        args.chunk, args.delta_t, not args.no_read, not args.no_write,
    )

    print("\n" + render(results, formats, args.metric, markdown=False))

    meta = dict(source=str(source), n_events=len(payload), formats=formats,
                libs=tokens, repeats=args.repeats, warmup=args.warmup,
                chunk=args.chunk, delta_t=args.delta_t, metric=args.metric)
    if args.markdown:
        Path(args.markdown).write_text(render(results, formats, args.metric, markdown=True) + "\n")
        print(f"\nwrote markdown -> {args.markdown}")
    if args.json:
        _dump_json(results, Path(args.json), meta)
        print(f"wrote json -> {args.json}")


if __name__ == "__main__":
    main()
