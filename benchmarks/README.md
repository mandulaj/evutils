# Benchmarks

Read/write throughput benchmarks for the native EVT2 / EVT2.1 / EVT3 codecs,
plus optional comparisons against other event libraries.

They are **not** part of the normal test run (`testpaths = ["tests"]`); run them
explicitly:

```bash
pytest benchmarks/                       # evutils only (+ any installed compare libs)
pytest benchmarks/test_read.py           # just reads
pytest benchmarks/test_write.py          # just writes
```

Fixtures (`test_events`, `real_event_files`) are shared with the test suite from
the repo-root `conftest.py`. `real_event_files` downloads the reference
recordings to the pytest cache on first use.

## Files

| file | what it benchmarks |
|------|--------------------|
| `test_read.py`          | evutils decode throughput (evt2/evt21/evt3), asserts count vs reference |
| `test_write.py`         | evutils encode throughput (payload = first 5M events of the real evt3 file) |
| `test_fixed_formats.py` | evutils read/write for DAT and AER (+ expelliarmus on DAT read) |
| `test_compare.py`       | third-party readers from `readers.py` (auto-skip if not installed) |
| `readers.py`            | adapter registry — one entry per external library |

DAT reuses the shared EVT3 reference events (its 14-bit coords fit 1280×720).
AER is 9-bit / timestamp-less (GenX320-class), so it uses a separate small
synthetic fixture.

## Comparing against other libraries

Install the optional readers and run with grouping so every library lines up per
format:

```bash
pip install evutils[compare]      # expelliarmus, evlib
pytest benchmarks/ --benchmark-group-by=param:fmt --benchmark-columns=mean,ops
```

Each library reads inside a lazy import, so an uninstalled (or broken) library
just **skips**. To add another library, append a `Reader(...)` to `readers.py`.

> tonic is intentionally not included: it has no standalone EVT reader and reads
> Prophesee data through `expelliarmus` internally, so it would just re-measure
> expelliarmus (already benchmarked).

## OpenEB / Metavision (via Docker)

OpenEB isn't on PyPI and is painful to build locally, so there's an image that
builds it once. Both commands run **from the repo root** (the build context must
be the whole project so evutils is copied in):

```bash
# Build the image (compiles OpenEB from source + installs evutils; slow, one-time)
docker build -t evutils-openeb -f benchmarks/docker/Dockerfile.openeb .

# Run the full suite (evutils + OpenEB), grouped per format
docker run --rm evutils-openeb
```

The container's default command is
`pytest benchmarks/ --benchmark-group-by=param:fmt`, so you get evutils and
OpenEB side by side per format.

### Useful variations

Persist the downloaded recording across runs (otherwise `--rm` discards the
pytest cache and it re-downloads every time):

```bash
docker run --rm -v evutils-cache:/work/.pytest_cache evutils-openeb
```

Run only the OpenEB comparison (skip the evutils rows):

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

- The image targets **OpenEB 5.x on Ubuntu 22.04**; OpenEB's apt dependencies
  and install layout drift between releases, so the `apt-get`/`PYTHONPATH` lines
  may need tweaking. The Dockerfile imports `metavision_core` at build time, so a
  broken OpenEB install fails during `docker build` rather than silently skipping
  at run time.
- The first pass is slow: it compiles OpenEB (`-j$(nproc)`) and downloads the
  reference recording on first run.
- A repo-root `.dockerignore` keeps the build context small (excludes `.venv`,
  `build/`, `.git`, `data/`, `*.raw`, ...).
