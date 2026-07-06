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
- [ ] **Random access / Timestamp indexing** (Big TODO for the future)
- [ ] **Arbitrary input sources:** memory-mapped IO, pure in-memory streams, HTTP streams
- [ ] **On-the-fly Compression wrappers:** passing file handles through `zstd` or `lz4` compression transparently before decoding

## Task Backlog

### High Priority
- [ ] **Random access / Timestamp indexing (`seek()`):** Implement timestamp-based random access via binary search. For EVT files, align to the first High Time, jump to the middle, parse until the next High Time to establish the time span, and binary search from there. For CSV, jump to the middle, find a newline, parse the next line's timestamp, and binary search. Support for DAT, NPZ, and HDF5 (if indexed). **Note:** This API must be restricted so it only works if both the format is seekable and the underlying `ByteSource` is seekable.
- [ ] **Documentation Expansion & Examples:** Go over all major functions (`EventReader`, `EventWriter`, `EventArray`, `SoaArray`) and add working `>>>` python examples directly into the docstrings. Additionally, create an `examples/` directory containing ready-written, standalone applications that demonstrate the core ideas and capabilities of the library.
- [ ] **Robust External Triggers Testing:** Bulletproof the reading and writing of external trigger data (especially in EVT2/3 formats) by implementing proper, comprehensive testing to guarantee perfect synchronization.

### Medium Priority
- [ ] **Transparent Compression Wrappers:** Allow `EventReader` to seamlessly read `.raw.zst` or `.csv.lz4` by wrapping the file handle in `zstandard` or `lz4` decoders, performing decompression on-the-fly directly into the C-parsers.
- [ ] **Comprehensive IO Testing:** Add real-file fixtures for `DAT`, `AER`, `AEDAT`, and `HDF5`. Implement "fuzzing/malformed data" tests to guarantee C-parser stability against corrupted byte streams without segfaulting.

### Completed
- [x] **Native C CSV Parser:** Eliminate the heavy `pandas` dependency by writing a heavily optimized native C parser for text-based event files.
- [x] **Real-time playback:** `EventReader(real_time=True, playback_speed=...)` paces `read()` and iteration against an absolute wall-clock anchor so chunks stream as if live; consumer processing time is absorbed automatically, and no delay is added when decoding falls behind.

### Future / Ongoing
- [ ] **Arbitrary input sources:** memory-mapped IO, pure in-memory streams, HTTP streams.
- [ ] **Performance Chasing:** Continuously benchmark against `evlib`, `expelliarmus`, and others, striving for the absolute fastest decoding times in the ecosystem.


