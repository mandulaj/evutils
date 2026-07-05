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
| `test_read.py`    | evutils decode throughput (evt2/evt21/evt3), asserts count vs reference |
| `test_write.py`   | evutils encode throughput (payload = first 5M events of the real evt3 file) |
| `test_compare.py` | third-party readers from `readers.py` (auto-skip if not installed) |
| `readers.py`      | adapter registry — one entry per external library |

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
builds it once:

```bash
docker build -t evutils-openeb -f benchmarks/docker/Dockerfile.openeb .
docker run --rm evutils-openeb
```

This runs the full benchmark suite (evutils + OpenEB) inside the container. See
the Dockerfile header for version/dependency caveats.
