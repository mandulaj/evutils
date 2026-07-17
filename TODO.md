# Development TODOs & Roadmap

## Core Philosophy
* **Fast & Lightweight:** Highly optimized C parsers for zero-bottleneck data ingestion.
* **Minimal Footprint:** Core features run entirely on NumPy and Numba.
* **Lazy Loading:** All heavy integrations (PyTorch, Pandas, HDF5, Polars, etc.) are lazy-loaded. If you don't use them, you don't need them installed, and they won't slow down import times.
* **Simple & Extensible:** Clean modular APIs.

## IO Goals & Features
We aim for universal event format support, prioritizing blazing fast read/write speeds, completeness, and extensibility. 
- [x] Universal format support (`.raw`, `.evt2`, `.dat`, `.aedat4`, `.hdf5`, `.npz`, `.csv`, etc.)
- [x] Full Read/Write parity where possible
- [x] Chunked & Streaming access
- [x] External trigger data parsing
- [ ] parquet 
- [x] **Random access / Timestamp indexing** (`EventReader.seek(t=/n=)`, by time or event index, forward/backward; hybrid strategy per format + optional Metavision `.tmp_index` sidecar)
- [x] **Arbitrary input sources:** memory-mapped IO, pure in-memory streams (HTTP streams pending)
- [ ] **On-the-fly Compression wrappers:** passing file handles through `zstd` or `lz4` compression transparently before decoding

## Task Backlog

### High Priority
- [ ] **AEDAT Support Shortcomings:** 
  - Extend the current `AEDAT` writer to fully support encoding for legacy AEDAT 1, 2, and 3 layouts (currently only AEDAT 4 is supported for writing). 
  - Implement parsers and data structures (e.g. `IMUArray`, `FrameArray` or integration via `DataBatch`) to extract and handle `IMUS` (IMU) and `FRME` (Frames) data streams from AEDAT 4.0 files.
  - Implement exact or index-based `seek()` functionality for `_aer.py` and `_aedat.py` (legacy AEDAT 1/2/3 and AER formats), which currently fall back to linear read skips.
- [ ] **Documentation Polish & Examples Cleanup:** Update `README.md` to reflect EVT4 support, fix quickstart snippets (e.g., `delta_t=10000`), fix docstring parameter orders (like `RPG_Reconstructor(height, width)`), and clean up scratch scripts in `examples/`.
- [ ] **EventStreamer Pipeline Refactor:** Decouple `EventReader`'s monolithic chunking logic into composable functional generators in `chunking.py`. Introduce a low-level `EventStreamer` for continuous byte-to-array decoding, and turn `EventReader` into a clean Façade that dynamically assembles these pipeline generators. This maintains backward-compatible ergonomics while allowing power-users to compose custom chunking pipelines (e.g., slicing by external trigger boundaries). *Status:* the `chunking.py` `stream_*` generators **exist but are not used by `EventReader`** and have drifted from its windowing/pacing/prefetch logic (they duplicate it). Do **not** treat them as the source of truth. Plan: extract per-mode `WindowStrategy` objects + a shared `SeekCursor` from `EventReader`, unify the generators onto those, then thin `EventReader` into a facade over them — so there is a single windowing implementation, not two.

### Medium Priority
- [ ] **Transparent Compression Wrappers:** Allow `EventReader` to seamlessly read `.raw.zst` or `.csv.lz4` by wrapping the file handle in `zstandard` or `lz4` decoders, performing decompression on-the-fly directly into the C-parsers.
- [x] **Comprehensive IO Testing:** Add real-file fixtures for `DAT`, `AER`, `AEDAT`, and `HDF5`. Implement "fuzzing/malformed data" tests to guarantee C-parser stability against corrupted byte streams without segfaulting.

### Completed
- [x] **Review bug-fix pass (2026-07-17):** Confirmed correctness bugs fixed + regression-tested (`tests/io/test_review_regressions.py`):
  1. EVT `seek()` now zeroes `_seek_correction` before its forward decode, so a second seek can't inherit the first's TIME_HIGH wrap correction.
  2. `jit.lazy_njit_unwrapped_events` selects `(t, x, y, p)` by field name (not `dtype.names` order), so dense kernels no longer scramble coordinates on non-canonical structured arrays.
  3. EVT3 C timestamp pipeline widened to 64-bit (state struct + `EMIT_SOA`), so timestamps survive past 2**32 µs (~71.6 min) on both scalar and vector events. *(DAT seek 32-bit wrap still open.)*
  4. `__builtin_unreachable()` on stray EVT3 packets removed. Corrupt-packet policy = robust default (warn + skip + resume) with `EventReader(strict=True)` to raise instead (SAFE vs UNRELIABLE).
  Plus: capability-flag contract centralized as base-`EventDecoder` defaults (`_exact_window`, `_independent_windows`, `_has_delta_t_parser`, `_use_sidecar`/`_raw_path`, `take_pending`) and read directly by `EventReader` (no more getattr/hasattr folklore silently degrading to the slow path); seek tests extended (evt4, normalize_ts semantics, repeated-seek-past-wrap, seek(n=)/seek(t=) past EOF, seek×triggers); C-parser fuzz test; README quickstart (`delta_t=10_000`) + `RPG_Reconstructor(width, height)` fixes; `test.raw` repo-root pollution + `test_evt3` ELF gitignored.
- [x] **Native C CSV Parser:** Eliminate the heavy `pandas` dependency by writing a heavily optimized native C parser for text-based event files.
- [x] **Real-time playback:** `EventReader(real_time=True, playback_speed=...)` paces `read()` and iteration against an absolute wall-clock anchor so chunks stream as if live; consumer processing time is absorbed automatically, and no delay is added when decoding falls behind.

### Future / Ongoing
- [x] **Arbitrary input sources:** memory-mapped IO, pure in-memory streams (HTTP streams pending).
- [ ] **Performance Chasing:** Continuously benchmark against `evlib`, `expelliarmus`, and others, striving for the absolute fastest decoding times in the ecosystem.

## ML & Computer Vision Ecosystem
To make `evutils` a one-stop-shop for training neural networks and running algorithms, the following features are planned:
- [ ] **Robust Noise Filtering:** Implement standard event stream cleanup algorithms, including Spatiotemporal (Background Activity) filters, Hot Pixel filters, and Refractory Period filters.
- [ ] **Spatial ML Augmentations (`transforms/`):** Build a library of composable geometric transformations for ML training, including Random Spatial/Center Crops, Random Flips (Horizontal/Vertical/Polarity), and Spatial Jitter.
- [ ] **PyTorch Integration (`torch/`):** Implement PyTorch-native versions of representations (e.g. `events_to_voxel_torch` to run on GPU) and `DataLoader` collators capable of batching variable-length event sequences efficiently.
- [ ] **Standardized Dataset APIs (`dataset/`):** Provide out-of-the-box downloaders and format wrappers for standard baselines (DVS128 Gesture, N-Cars, MVSEC, 1 Megapixel Automotive Dataset) to eliminate data loading boilerplate.
- [ ] **Algorithmic Baselines:** Implement traditional CV algorithm baselines directly on events, such as Contrast Maximization for evaluating optical flow and motion compensation data quality.
