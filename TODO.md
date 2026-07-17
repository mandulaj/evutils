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
- [ ] **AEDAT Encoder Parity & Features:** Extend the current `AEDAT` writer to fully support encoding for legacy AEDAT 1, 2, and 3 layouts (currently only AEDAT 4 is supported for writing). Additionally, implement parsers and data structures (e.g. `IMUArray`, `FrameArray` or integration via `DataBatch`) to extract and handle `IMUS` (IMU) and `FRME` (Frames) data streams from AEDAT 4.0 files.
- [x] **Random access / Timestamp indexing (`seek()`):** `EventReader.seek(t=/n=, relative=)` repositions by absolute timestamp or event index, forward or backward; reads then continue in the configured mode. Per-format strategy: DAT exact record math; EVT via a `SeekIndex` (an exact in-memory build by default, lazy on first seek; a Metavision `.tmp_index` sidecar is opt-in via `index="metavision"` -- fast but approximate near large event gaps) with a TIME_HIGH wrap correction; CSV byte-offset binary search + newline resync; NPZ/HDF5 index/searchsorted. Falls back to iterate-and-skip on non-seekable sources. *Remaining:* make the Metavision sidecar path gap-exact (currently the built index is exact, sidecar can be off by up to ~one TIME_LOW near gaps); persist our own index sidecar; preserve triggers across the seek boundary chunk.
- [x] **Documentation Expansion & Examples:** Go over all major functions (`EventReader`, `EventWriter`, `EventArray`, `SoaArray`) and add working `>>>` python examples directly into the docstrings. Additionally, create an `examples/` directory containing ready-written, standalone applications that demonstrate the core ideas and capabilities of the library.
- [x] **Robust External Triggers Testing:** Bulletproof the reading and writing of external trigger data (especially in EVT2/3 formats) by implementing proper, comprehensive testing to guarantee perfect synchronization.
- [ ] **EventStreamer Pipeline Refactor:** Decouple `EventReader`'s monolithic chunking logic into composable functional generators in `chunking.py`. Introduce a low-level `EventStreamer` for continuous byte-to-array decoding, and turn `EventReader` into a clean Façade that dynamically assembles these pipeline generators. This maintains backward-compatible ergonomics while allowing power-users to compose custom chunking pipelines (e.g., slicing by external trigger boundaries). *(`EventStreamer` and `chunking.py` generators are complete; `EventReader` Facade integration is pending).*

### Medium Priority
- [ ] **Transparent Compression Wrappers:** Allow `EventReader` to seamlessly read `.raw.zst` or `.csv.lz4` by wrapping the file handle in `zstandard` or `lz4` decoders, performing decompression on-the-fly directly into the C-parsers.
- [x] **Comprehensive IO Testing:** Add real-file fixtures for `DAT`, `AER`, `AEDAT`, and `HDF5`. Implement "fuzzing/malformed data" tests to guarantee C-parser stability against corrupted byte streams without segfaulting.

### Completed
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
