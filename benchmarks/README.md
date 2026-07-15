# Benchmarks

Read/write **throughput** benchmarks (M events/s) for the native codecs
(evt3 / evt2 / evt21 / evt4 / dat / csv), plus optional comparisons against other
event libraries.

Everything is driven by a single script, **`benchmarks/throughput.py`**. It is
not part of the normal test run and must be executed explicitly from the repo
root.

## Quick start

```bash
# evutils only + any installed comparison lib; downloads the 'normal' dataset once
python benchmarks/throughput.py

# fast smoke run
python benchmarks/throughput.py --dataset small --events 2_000_000 --repeats 2

# benchmark the payload decoded from a specific local recording
python benchmarks/throughput.py --file data/hand.raw
```

## What it measures

The script produces **two matrices** — rows are formats, columns are
library/variant, cells are **M events/s** (higher is better):

- **Read matrix.** The evutils read path is split into the variants whose cost we
  want visible separately:
  - `evutils n_events` — `EventReader` slicing by a fixed event count (the common
    chunked-read path);
  - `evutils delta_t` — `EventReader` slicing by a fixed time window. The gap to
    `n_events` on the same row **is** the delta_t slicing penalty (an extra
    `searchsorted` per window);
  - `evutils async` — `EventReader(async_read=True)`, background-thread prefetch;
  - `evutils stream` — `EventStreamer`, raw chunked decode with no slicing: the
    throughput **ceiling** and the true peer of `expelliarmus`'s `read_chunk`.

  plus a column per installed comparison library (see below).

- **Write matrix.** `evutils` vs any installed comparison writer.

## Methodology

- **In RAM.** Every file lives on a **tmpfs / RAM disk** (default `/dev/shm`), so
  disk I/O never enters the measurement — after warmup the file is served from
  the page cache at RAM speed. This applies to *every* library, so the
  comparison is fair.
- **Chunked only, never `read_all`.** Real recordings don't fit in memory;
  production code always streams. So does the benchmark.
- **One real event set for every format.** Real recordings ship only as
  evt2/evt3/evt21, so the script decodes the first `--events N` events of one
  recording into a single in-RAM array (the decoded array is what can't fit
  whole, hence the cap), then — one format at a time to bound RAM/disk — writes
  that identical set to the RAM disk and benchmarks reads + writes on it. Same
  events, same count across all formats ⇒ the numbers are directly comparable.
- **Timing.** Hand-rolled: `--warmup` runs, then `--repeats` timed runs;
  `gc.collect()` between; report **peak** (fastest run) and **mean**. No
  pytest-benchmark, no extra deps.
- **Validation.** Each read cell checks the decoded count against the payload
  size and flags a mismatch with `!!` in the cell.

## Comparison libraries

Enabled via `--libs` (prefix match; `evutils` selects all four evutils variants):

| column | package | formats | notes |
|--------|---------|---------|-------|
| `expelliarmus` | `expelliarmus` | evt2, evt3, dat | `read_chunk` (read) + `save` (write) |
| `evlib`        | `evlib` (Rust) | evt2, evt3 | streaming `collect`, read only |
| `evt3`         | `evt3` (Rust)  | evt3 | `decode_file`, one-shot read only |
| `openeb`       | Metavision SDK | evt2, evt21, evt3 | `RawReader`; Docker only (see below) |

```bash
pip install evutils[compare]                        # expelliarmus, evlib
python benchmarks/throughput.py --libs evutils,expelliarmus,evlib,evt3
```

Each adapter lazy-imports its library, so a missing/broken install just makes
that **column disappear** (never an error). Unsupported (format, library) cells
render `-`. To add a library, append one entry to `READ_ADAPTERS` /
`WRITE_ADAPTERS` in `throughput.py`.

> `tonic` is intentionally excluded: it has no standalone EVT reader and reads
> Prophesee data through `expelliarmus` internally, so it would just re-measure
> `expelliarmus`.

## Options

| flag | default | meaning |
|------|---------|---------|
| `--dataset {small,normal,large}` | `normal` | tier to download (cached, shared with the test suite) |
| `--file PATH` | — | decode the payload from this local recording instead |
| `--events N` | 20M | events decoded into the in-RAM payload |
| `--formats` | `evt3,evt2,evt21,evt4,dat,csv` | formats to benchmark |
| `--containers` | off | also benchmark `npz,hdf5,aer` |
| `--libs` | `evutils,expelliarmus,evlib` | libraries to enable (add `evt3,openeb`) |
| `--repeats` / `--warmup` | 5 / 1 | timed runs / warmup runs per cell |
| `--chunk` | 1M | read chunk size (n_events / async / openeb) |
| `--delta-t` | 10000 | time window (µs) for the delta_t variant |
| `--tmpfs` | `/dev/shm` | RAM-disk directory |
| `--metric {peak,mean}` | `peak` | metric shown in the matrices |
| `--markdown PATH` / `--json PATH` | — | also write the matrices as markdown / full stats as JSON |
| `--no-read` / `--no-write` | — | skip a phase |

Set `EVUTILS_BENCH_DATA` to a directory of already-extracted recordings (+ JSON
sidecars) to skip the download entirely (offline / Docker).

## Regenerating the docs table

```bash
python benchmarks/throughput.py --markdown docs/source/benchmarks/results.md
```

## OpenEB / Metavision (via Docker)

OpenEB isn't on PyPI and is fiddly to build, so there's an image that builds it
once. Run **from the repo root** (the build context must be the whole project):

```bash
docker build -t evutils-openeb -f benchmarks/docker/Dockerfile.openeb .
docker run --rm evutils-openeb            # evutils + expelliarmus + evlib + openeb
```

The container's default command runs `throughput.py` with `--libs
...,openeb`, so the `openeb` column is populated for evt2/evt21/evt3.

The recordings are not baked into the image; on first run the script downloads
them. For an offline container, mount the host cache and point
`EVUTILS_BENCH_DATA` at it:

```bash
docker run --rm \
  -v "$(pwd)/.pytest_cache/d/event_files:/data:ro" \
  -e EVUTILS_BENCH_DATA=/data \
  evutils-openeb
```

Drop into a shell to debug:

```bash
docker run --rm -it evutils-openeb bash
```

### Caveats

- The image targets **OpenEB 5.x on Ubuntu 22.04**; OpenEB's apt deps and layout
  drift between releases, so the `apt-get`/`PYTHONPATH` lines may need tweaking.
  The Dockerfile imports `metavision_core` at build time, so a broken install
  fails the build rather than silently skipping.
- The first pass is slow: it compiles OpenEB (`-j$(nproc)`) and downloads the
  reference recording.

## Other benchmark tools

- `benchmarks/decode_micro.py` — isolates the raw **C decoder** inner loop (warm
  reused buffer) from streaming/Python overhead; complementary to this script.
- `benchmarks/legacy/` — the retired pytest-benchmark suite, kept for reference
  (see its README). Not maintained.
