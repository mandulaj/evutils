# Benchmarks

Read/write throughput benchmarks for the native EVT2 / EVT2.1 / EVT3 codecs, plus optional comparisons against other event libraries.

These benchmarks are **not** part of the normal test run (which uses `testpaths = ["tests"]`). They must be executed explicitly from the repository root.

## Quick Start

Run the default suite (evutils only, plus any installed comparison libraries):
```bash
pytest benchmarks/
```

Run specific benchmarks:
```bash
pytest benchmarks/test_read.py           # just reads
pytest benchmarks/test_write.py          # just writes
```

## Benchmark Options

### Datasets
You can benchmark against two datasets using the `--dataset` flag. The necessary files are automatically downloaded and cached on first use.
- `--dataset small` (default): A ~1GB memory footprint dataset (hand recordings).
- `--dataset large`: A massive multi-GB dataset designed to test the streaming and chunking capabilities of the decoders.

```bash
pytest benchmarks/ --dataset large
```

### Filtering Readers
The benchmark tests are fully parametrized by the reader name. You can effortlessly exclude specific third-party libraries using pytest's standard `-k` flag (for example, if they are slow or misbehaving on large datasets):

```bash
pytest benchmarks/ -k "not evlib" --dataset large
pytest benchmarks/ -k "not evlib and not openeb"
```

## File Structure

| File | What it benchmarks |
|------|--------------------|
| `test_read.py`          | `evutils` decode throughput on the full real recordings (evt2/evt21/evt3), asserts count vs reference |
| `test_write.py`         | `evutils` encode throughput (payload = first 5M events of the real evt3 file) |
| `test_formats.py`       | **uniform per-format read/write**: the same 5M real events transcoded into every format (EVT3/EVT2.1/EVT2, DAT, AER, NPZ, HDF5, CSV, AEDAT4), so numbers are comparable across formats; expelliarmus (DAT) and evlib (HDF5, AEDAT4) compared on the identical files |
| `test_compare.py`       | third-party readers from `readers.py` on the full real recordings (auto-skip if not installed) |
| `readers.py`            | adapter registry — one entry per external library |

*Note: in `test_formats.py` AER is the one lossy transcode — the format has no timestamps and 9-bit coordinates, so the same events are written with coordinates masked to 0–511 (identical event count).*

## Comparing Against Other Libraries

Install the optional readers and run with grouping so every library lines up per format:

```bash
pip install evutils[compare]      # expelliarmus, evlib
pytest benchmarks/ --benchmark-group-by=param:fmt --benchmark-columns=mean,ops
```

Each library reads inside a lazy import. If a library is uninstalled or broken, its benchmarks simply **skip**. To add another library, append a `Reader(...)` entry to `readers.py`.

## Benchmark Comparison (Mean Time in Seconds)

### Reading

| Library | EVT2 | EVT21 | EVT3 |
|---|---|---|---|
| **evutils** | 0.138 s | 0.070 s | 0.292 s |
| **evlib** | 4.410 s | N/A | 4.327 s |
| **expelliarmus** | 0.128 s | N/A | 0.341 s |

### Writing

| Library | EVT2 | EVT21 | EVT3 |
|---|---|---|---|
| **evutils** | 0.013 s | 0.028 s | 0.041 s |
| **expelliarmus** | 0.077 s | N/A | 0.097 s |

**Hardware:** 12th Gen Intel(R) Core(TM) i7-1280P | **OS:** Linux 7.1.1-3-MANJARO | **Python:** 3.12.13

*Lower is better. Generated dynamically by `scripts/generate_benchmark_table.py`.*

> **Note**: `tonic` is intentionally not included. It has no standalone EVT reader and reads Prophesee data through `expelliarmus` internally, so benchmarking it would just re-measure `expelliarmus`.

## OpenEB / Metavision (via Docker)

OpenEB isn't on PyPI and is painful to build locally, so there's an image that builds it once. Both commands must be run **from the repo root** (the build context must be the whole project so `evutils` is copied in):

```bash
# Build the image (compiles OpenEB from source + installs evutils; slow, one-time)
docker build -t evutils-openeb -f benchmarks/docker/Dockerfile.openeb .

# Run the full suite (evutils + OpenEB), grouped per format
docker run --rm evutils-openeb
```

The container's default command is `pytest benchmarks/ --benchmark-group-by=param:fmt`, so you get `evutils` and OpenEB side by side per format.

### Useful Variations

Persist the downloaded recording across runs (otherwise `--rm` discards the pytest cache and it re-downloads every time):

```bash
docker run --rm -v evutils-cache:/work/.pytest_cache evutils-openeb
```

Run only the OpenEB comparison (skip the `evutils` rows):

```bash
docker run --rm evutils-openeb \
  pytest benchmarks/test_compare.py --benchmark-group-by=param:fmt -q
```

Drop into a shell to debug the build/run:

```bash
docker run --rm -it evutils-openeb bash
```

Build a different OpenEB release if the default fails to build:

```bash
docker build -t evutils-openeb --build-arg OPENEB_VERSION=5.0.0 \
  -f benchmarks/docker/Dockerfile.openeb .
```

### Caveats

- The image targets **OpenEB 5.x on Ubuntu 22.04**; OpenEB's apt dependencies and install layout drift between releases, so the `apt-get`/`PYTHONPATH` lines may need tweaking. The Dockerfile imports `metavision_core` at build time, so a broken OpenEB install fails during `docker build` rather than silently skipping at run time.
- The first pass is slow: it compiles OpenEB (`-j$(nproc)`) and downloads the reference recording on first run.
- A repo-root `.dockerignore` keeps the build context small (excludes `.venv`, `build/`, `.git`, `data/`, `*.raw`, etc.).
