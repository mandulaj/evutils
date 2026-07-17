#!/usr/bin/env python3
"""Isolated C-decoder micro-benchmark (evt2 / evt21 / evt4 / dat / csv).

Why this exists
---------------
"Decode throughput" measured end to end swings ~3x purely on buffer strategy,
not on the decoder:

* a single decode into one big fresh output buffer is **fault-bound** (the
  output is cold memory) -- e.g. evt2 ~220 M events/s;
* EventStreamer reuses the decoder's buffer but adds per-chunk Python overhead
  -- ~600 M events/s;
* the C loop alone, decoding in chunks into a **reused warm** buffer, is the
  actual decoder speed -- ~680 M events/s.

To test changes to the C decoders you want the last number: this script drives
``parse_chunk_soa`` (or ``evutils_read_csv``) directly in a loop over a reused
buffer, so it measures the decode inner loop with no streaming/allocation noise.

Usage
-----
    python benchmarks/decode_micro.py                       # synthetic, defaults
    python benchmarks/decode_micro.py --events 20_000_000 --formats evt2,dat,csv
    python benchmarks/decode_micro.py --file rec.raw        # one real file
"""
from __future__ import annotations

import argparse
import ctypes
import gc
import os
import time
from pathlib import Path
from typing import Any

import numpy as np

from evutils.io import EventWriter
from evutils.io._native_core import EventSoABuffers, TriggerSoABuffers, lib
from evutils.io._source import make_source
from evutils.random import random_events

# Binary formats: name -> (Input class, Parser class, numpy word dtype).
from evutils.io._native_evt import (
    Evt2Input, Evt2Parser, Evt21Input, Evt21Parser, Evt4Input, Evt4Parser,
)
from evutils.io._native_dat import DatInput, DatParser

_BINARY = {
    "evt2": (Evt2Input, Evt2Parser, np.uint32, ".raw"),
    "evt21": (Evt21Input, Evt21Parser, np.uint64, ".raw"),
    "evt4": (Evt4Input, Evt4Parser, np.uint32, ".raw"),
    "dat": (DatInput, DatParser, np.uint32, ".dat"),
}
# evt3's vector groups need look-ahead tail padding on the final chunk, which the
# simple reused-loop here doesn't emulate; benchmark it via throughput.py.


def _payload_words(path: Path, word_dtype: Any, is_dat: bool) -> np.ndarray:
    """Return the binary payload past the header as a contiguous word array."""
    from evutils.io._evt import EventDecoder_EVT
    from evutils.io._dat import EventDecoder_Dat

    dec = (EventDecoder_Dat if is_dat else EventDecoder_EVT)(make_source(str(path)))
    dec.init()
    words = np.ascontiguousarray(dec._words)
    dec.close()
    return words


def _time(fn: Any, repeats: int, warmup: int) -> tuple[float, float, int]:
    n = 0
    for _ in range(warmup):
        n = fn()
        gc.collect()
    ts = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        n = fn()
        ts.append(time.perf_counter() - t0)
        gc.collect()
    return min(ts), sum(ts) / len(ts), n


def bench_binary(words: np.ndarray, fmt: str, chunk: int, repeats: int, warmup: int) -> int:
    InputCls, ParserCls, _dtype, _ext = _BINARY[fmt]
    parser = ParserCls()
    ev = EventSoABuffers(chunk + 8192)   # one reused, warm output buffer
    tr = TriggerSoABuffers(8192)
    nw = len(words)

    def decode() -> int:
        parser.reset()
        off = 0
        total = 0
        while off < nw:
            ev.reset()
            inp = InputCls(words[off:])
            res = parser.parse_chunk_soa(inp, ev, tr)
            consumed = inp.consumed(res)
            if consumed <= 0:
                break
            off += consumed
            total += ev.size
        return total

    mn, mean, n = _time(decode, repeats, warmup)
    _report(fmt, n, mn, mean)
    return n


def bench_csv(payload: bytes, chunk: int, repeats: int, warmup: int) -> int:
    # Reused output columns (t,x,y,p); decode the buffer in chunk-event slices.
    t = np.empty(chunk, np.int64)
    x = np.empty(chunk, np.uint16)
    y = np.empty(chunk, np.uint16)
    p = np.empty(chunk, np.uint8)
    outs = (ctypes.c_void_p * 4)(t.ctypes.data, x.ctypes.data, y.ctypes.data, p.ctypes.data)
    types = (ctypes.c_int * 4)(8, 2, 2, 1)
    cmap = (ctypes.c_int * 4)(0, 1, 2, 3)
    bc = ctypes.c_size_t()
    ep = ctypes.c_size_t()
    # numpy view over the bytes so slicing is zero-copy (bytes[off:] would copy
    # the whole tail each chunk); pass base+off as the C buffer pointer.
    arr = np.frombuffer(payload, dtype=np.uint8)
    total_len = arr.nbytes
    base = arr.ctypes.data

    def decode() -> int:
        off = 0
        total = 0
        while off < total_len:
            ptr = ctypes.cast(base + off, ctypes.c_char_p)
            res = lib().evutils_read_csv(ptr, total_len - off, b","[0], outs, types, cmap, 4,
                                   chunk, ctypes.byref(ep))
            if ep.value == 0:
                break
            off += (res.current - ctypes.cast(ptr, ctypes.c_void_p).value) if res.current is not None else 0
            total += ep.value
        return total

    mn, mean, n = _time(decode, repeats, warmup)
    _report("csv", n, mn, mean)
    return n


def _report(fmt: str, n: int, mn: float, mean: float) -> None:
    peak = n / mn / 1e6 if mn else float("nan")
    avg = n / mean / 1e6 if mean else float("nan")
    print(f"  {fmt:6s} decode  peak {peak:8.1f}  mean {avg:8.1f} Mev/s   ({n:,} events)")


def _csv_payload(path: Path) -> bytes:
    """Bytes of a .csv past its header line."""
    raw = path.read_bytes()
    nl = raw.find(b"\n")
    return raw[nl + 1:] if nl >= 0 else raw


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--events", type=int, default=20_000_000)
    ap.add_argument("--formats", type=str, default="evt2,evt21,evt4,dat,csv")
    ap.add_argument("--chunk", type=int, default=1_000_000)
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--tmpfs", type=str, default="/dev/shm")
    ap.add_argument("--file", type=str, default=None,
                    help="decode ONE real file (its format only)")
    args = ap.parse_args()

    tmpdir = Path(args.tmpfs)

    if args.file:
        from evutils.io._evt import EventDecoder_EVT  # noqa: F401 (import cost warm-up)
        path = Path(args.file)
        fmt = _detect(path)
        print(f"decode (reused warm buffer) -- real file {path.name} [{fmt}]")
        if fmt == "csv":
            bench_csv(_csv_payload(path), args.chunk, args.repeats, args.warmup)
        elif fmt in _BINARY:
            words = _payload_words(path, _BINARY[fmt][2], fmt == "dat")
            bench_binary(words, fmt, args.chunk, args.repeats, args.warmup)
        else:
            raise SystemExit(f"decode_micro does not cover {fmt}")
        return

    formats = [f.strip() for f in args.formats.split(",") if f.strip()]
    print(f"decode (reused warm buffer) -- {args.events/1e6:.0f} M synthetic events")
    data = random_events(args.events)
    for fmt in formats:
        if fmt == "csv":
            path = tmpdir / "dm.csv"
            with EventWriter(path) as w:
                w.write(data)
            bench_csv(_csv_payload(path), args.chunk, args.repeats, args.warmup)
            path.unlink(missing_ok=True)
        elif fmt in _BINARY:
            ext = _BINARY[fmt][3]
            path = tmpdir / f"dm_{fmt}{ext}"
            with (EventWriter(path) if fmt == "dat" else EventWriter(path, format=fmt)) as w:
                w.write(data)
            words = _payload_words(path, _BINARY[fmt][2], fmt == "dat")
            path.unlink(missing_ok=True)
            bench_binary(words, fmt, args.chunk, args.repeats, args.warmup)
        else:
            print(f"  {fmt:6s} skipped (not covered; use throughput.py for evt3)")
        gc.collect()


def _detect(path: Path) -> str:
    from evutils.io.decoders import resolve_decoder_cls
    src = make_source(str(path))
    cls = resolve_decoder_cls(src)
    dec = cls(src)
    dec.init()
    fmt = getattr(dec, "_format", None)
    dec.close()
    if fmt:
        return str(fmt)
    name = cls.__name__.lower()
    for cand in ("dat", "csv"):
        if cand in name:
            return cand
    raise SystemExit(f"unsupported format for {path}")


if __name__ == "__main__":
    main()
