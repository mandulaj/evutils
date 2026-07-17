# Development TODOs & Roadmap

## Core Philosophy
* **Fast & Lightweight:** Highly optimized C parsers for zero-bottleneck data ingestion.
* **Minimal Footprint:** Core features run entirely on NumPy and Numba.
* **Lazy Loading:** All heavy integrations (PyTorch, Pandas, HDF5, Polars, etc.) are lazy-loaded. If you don't use them, you don't need them installed, and they won't slow down import times.
* **Simple & Extensible:** Clean modular APIs.

## IO Goals & Features
We aim for universal event format support, prioritizing blazing fast read/write speeds, completeness, and extensibility.
(Completed goals are pruned from this file — the README's Goals & Roadmap keeps the full checklist.)
- [ ] parquet support
- [ ] `.bin` format (currently a reserved `NotImplementedError` stub in `io/_bin.py`)
- [ ] **On-the-fly Compression wrappers:** read `.raw.zst` / `.csv.lz4` seamlessly by wrapping the file handle in `zstandard` / `lz4` decoders, decompressing on-the-fly directly into the C-parsers.

## Task Backlog

### High Priority
- [ ] **AEDAT Support Shortcomings:**
  - Extend the current `AEDAT` writer to fully support encoding for legacy AEDAT 1, 2, and 3 layouts (currently only AEDAT 4 is supported for writing).
  - Implement parsers and data structures (e.g. `IMUArray`, `FrameArray` or integration via `DataBatch`) to extract and handle `IMUS` (IMU) and `FRME` (Frames) data streams from AEDAT 4.0 files.
  - Implement exact or index-based `seek()` functionality for `_aer.py` and `_aedat.py` (legacy AEDAT 1/2/3 and AER formats), which currently fall back to linear read skips.
- [ ] **EventStreamer Pipeline Refactor:** Decouple `EventReader`'s monolithic chunking logic into composable functional generators in `chunking.py`. Introduce a low-level `EventStreamer` for continuous byte-to-array decoding, and turn `EventReader` into a clean Façade that dynamically assembles these pipeline generators. This maintains backward-compatible ergonomics while allowing power-users to compose custom chunking pipelines (e.g., slicing by external trigger boundaries). *Status:* the `chunking.py` `stream_*` generators **exist but are not used by `EventReader`** and have drifted from its windowing/pacing/prefetch logic (they duplicate it: `stream_paced_playback` lacks `max_gap`, `stream_n_events` cuts triggers inconsistently). Do **not** treat them as the source of truth. Plan: extract per-mode `WindowStrategy` objects + a shared `SeekCursor` from `EventReader`, unify the generators onto those, then thin `EventReader` into a facade over them — so there is a single windowing implementation, not two.

### Medium Priority
- [ ] **Seek polish:**
  - Make the Metavision `.tmp_index` sidecar path gap-exact (the built index is exact; the sidecar can be off by up to ~one TIME_LOW near large event gaps).
  - Persist our own index as a sidecar so repeated opens skip the lazy build.
  - Unify `SeekResult.index` conventions (CSV time-seek returns `-1`, so `len(reader)` / relative `seek(n=)` restart from 0 after it).
  - Unify `tell()` semantics across decoders (byte offset for EVT/DAT/AER vs event index for NPZ/HDF5).
- [ ] **CSV decoder cleanup:** header detection requires a seekable source (`readline()` + `seek(0)` at init, live TODO in `_csv.py`); `read_chunk` asserts instead of auto-`init()` like every other decoder.
- [ ] **Lazy vis imports:** `evutils.vis` eagerly imports `plot3d` (cv2 + matplotlib) and `reconstructor/_base.py` imports torch at module top — contradicts the lazy-loading philosophy; defer via module `__getattr__` like the top-level package.
- [ ] **C parser debt:** `EVUTILS_PARSE_ERROR` is never emitted by any parser; `EVUTILS_PARSE_WARNING` (corrupt-packet skip) exists only in EVT3 — extend to EVT2/2.1/4/DAT; duplicate unused `evt3_parse_status_t` enum + stale "ACTION REQUIRED" comment in `evt3.h`; `EVT3_parse_delta_t_soa` duplicates ~130 lines of `parse_chunk_soa`.
- [ ] **Test gaps:** `reuse_buffers` aliasing-contract test; `EventReader(start_ts=, max_time=)`, `tell()`, `__len__` untested; consolidate the ~9 duplicated event-generator helpers into `conftest_utils`; remove the half-dead `event_files` fixture (`tests/io/conftest.py`).

### Future / Ongoing
- [ ] **Performance Chasing:** Continuously benchmark against `evlib`, `expelliarmus`, and others, striving for the absolute fastest decoding times in the ecosystem. (Includes the dynamic CPU-dispatch idea noted in `csrc/evt3.c`.)
- [ ] **Documentation depth:** narrative docs pages for the flagship features (seek/random access, async_read/reuse_buffers, transforms, dense); decide the fate of the untracked scratch scripts in `examples/` (`ev.py`, `exp.py`, `mv.py`, `read_bench.py`, `stream.py` — salvage the pacing/buffer-ring commentary, then delete or polish into numbered examples).

## ML & Computer Vision Ecosystem
To make `evutils` a one-stop-shop for training neural networks and running algorithms, the following features are planned:
- [ ] **Robust Noise Filtering:** Implement standard event stream cleanup algorithms, including Spatiotemporal (Background Activity) filters, Hot Pixel filters, and Refractory Period filters (a refractory functional exists in `transforms/`; `filtering/` has only spatial masking).
- [ ] **Spatial ML Augmentations (`transforms/`):** flips LR, spatial/time jitter, drops and refractory exist; still missing: Random Spatial/Center Crops, vertical flip, polarity flip — and crops must update the `sensor_size` metadata. Also: replace global legacy `np.random` seeding with a passable `numpy.random.Generator` for per-worker reproducibility, and validate ctor params (`DropEventByTime` ratio, `TimeJitter` std).
- [ ] **Dense representations polish:** consume `events.sensor_size` metadata instead of hardcoded 1280x720 defaults; `tore()`'s `dtype` parameter is accepted but unused.
- [ ] **PyTorch Integration (`torch/`):** stub — implement PyTorch-native representations (e.g. `events_to_voxel_torch` on GPU) and `DataLoader` collators capable of batching variable-length event sequences efficiently.
- [ ] **Standardized Dataset APIs (`dataset/`):** stub — provide out-of-the-box downloaders and format wrappers for standard baselines (DVS128 Gesture, N-Cars, MVSEC, 1 Megapixel Automotive Dataset) to eliminate data loading boilerplate.
- [ ] **Algorithmic Baselines:** Implement traditional CV algorithm baselines directly on events, such as Contrast Maximization for evaluating optical flow and motion compensation data quality.
