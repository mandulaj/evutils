# <a href="https://mandulaj.github.io/evutils"><img src="https://mandulaj.github.io/evutils/_static/event_hexagon_broken.webp" alt="evutils_logo" width="50" align="top" style="background-color: #fff0;"></a> EV-Utils
[![PyPI Version](https://img.shields.io/pypi/v/evutils)](https://pypi.org/project/evutils/)
[![PyPI Python Version](https://img.shields.io/pypi/pyversions/evutils)](https://pypi.org/project/evutils/)
[![Release & Publish Docs](https://github.com/mandulaj/evutils/actions/workflows/release.yaml/badge.svg)](https://github.com/mandulaj/evutils/actions/workflows/release.yaml)
[![Test](https://github.com/mandulaj/evutils/actions/workflows/test.yaml/badge.svg)](https://github.com/mandulaj/evutils/actions/workflows/test.yaml)
[![Coverage](https://mandulaj.github.io/evutils/coverage/badge.svg)](https://mandulaj.github.io/evutils/coverage/)
[![Documentation](https://img.shields.io/badge/docs-GitHub%20Pages-blue)](https://mandulaj.github.io/evutils/)
[![GitHub code size](https://img.shields.io/github/languages/code-size/mandulaj/evutils)](https://github.com/mandulaj/evutils)
[![GitHub License](https://img.shields.io/github/license/mandulaj/evutils)](https://github.com/mandulaj/evutils?tab=GPL-3.0-1-ov-file)


## Overview
EV-Utils (`evutils`) is a performant collection of utilities for working with event-based vision data. Built with minimal dependencies, it relies on a compiled C backend for speed while offering a clean, modular Python interface.

### Core Philosophy
* **Fast & Lightweight:** Highly optimized C parsers for zero-bottleneck data ingestion.
* **Minimal Footprint:** Core features run entirely on NumPy and Numba.
* **Lazy Loading:** All heavy integrations (PyTorch, HDF5, etc.) are lazy-loaded. If you don't use them, you don't need them installed, and they won't slow down import times.
* **Simple & Extensible:** Clean modular APIs.



### Inspirations & Related Work 
This project draws inspiration from several excellent libraries in the event-based vision ecosystem and attempts to fill in their shortcomings:

* [Tonic](https://github.com/neuromorphs/tonic)
* [Faery](https://github.com/aestream/faery)
* [event_utils](https://github.com/TimoStoff/event_utils)
* [evlib](https://github.com/tallamjr/evlib)
* [expelliarmus](https://github.com/open-neuromorphic/expelliarmus)
* [event-vision-library](https://github.com/shiba24/event-vision-library)
* [evt3](https://github.com/muthmann/evt3)
* [openeb](https://github.com/prophesee-ai/openeb)


## Installation
We recommend installing `evutils` using `uv`. 
### From PyPi
```bash
uv add evutils # Basic library
uv add evutils[all] # All groups (torch, hdf5, aedat, vis, etc..)
uv add evutils[dev] # Dev group
```

### From Git
```bash
git clone --recurse-submodules https://github.com/mandulaj/evutils.git
cd evutils

uv pip install -e ".[dev]"
```

Note: You can also install specific optional dependency groups like `uv add evutils[torch,hdf5]`.

## Architecture
The library is divided into several discrete modules. Many can be used independently without installing the full suite of dependencies:

```
└── augment     - Event augmentations
└── chunking    - Splitting event streams into fixed-size windows
└── dataset     - Wrappers for various dataset loaders
└── dense       - Dense representations (voxel grids, time surfaces, histograms)
└── filtering   - Event stream filtering (masking, timestamp normalization)
└── io          - Event reading and writing interfaces
    ├── reader 
    └── writer
└── random      - Random event generation and noise injection
└── torch       - PyTorch integration (requires evutils[torch])
└── transforms  - torchvision-style augmentation transforms (+ functional)
└── types       - Standard types for representing Events in NumPy arrays
└── vis         - Visualization methods
    ├── histogram
    └── reconstructor
```

A future `sparse` module will sit alongside `dense` once sparse representations
(event graphs, sparse tensors, point clouds) land. `EventsChecker` (event-array
validation) lives in `types`.

### Quick API overview 

<!-- ### `augment`

Event augmentations

### `dataset`

Wrappers for various dataset loaders -->

#### `io`: Reading and Writing Events

The `io` module provides methods for reading and writing events into various event formats. It provides a simple `.read()` and `.write()` interface as well as more advanced interfaces using iterators and slicing.

Supported formats (see the [formats documentation](https://mandulaj.github.io/evutils/formats.html) for details):

| Format | Extensions | Read | Write | Notes |
|---|---|:---:|:---:|---|
| EVT3 / EVT2.1 / EVT2 (Prophesee RAW) | `.raw`, `.evt*` | ✅ | ✅ | native C decoder, external triggers |
| DAT (Prophesee) | `.dat` | ✅ | ✅ | native C decoder |
| AER (Prophesee) | `.aer` | ✅ | ✅ | timestamp generation selectable |
| AEDAT 1.0 / 2.0 / 3.1 / 4.0 | `.aedat`, `.aedat4` | ✅ | 🚧 | AEDAT4 compression: `evutils[aedat]` |
| HDF5 (DSEC/RVT layout) | `.h5`, `.hdf5` | ✅ | ✅ | `evutils[hdf5]`, ms-index random access |
| HDF5 (Prophesee layout) | `.h5`, `.hdf5` | ✅ | 🚧 | ECF-compressed files need the ECF plugin |
| NPZ | `.npz` | ✅ | ✅ | streaming, `np.load`-compatible |
| CSV / TXT | `.csv`, `.txt` | ✅ | ✅ | native C parser |
| BIN | `.bin` | 🚧 | 🚧 | planned |

```python
from evutils.io import EventReader


ev_file = EventReader("raw_file.raw", delta_t=10e3)

events = ev_file.read()

```

It also supports **random access** — jump to an absolute timestamp or event
index (forward or backward) and keep reading in the configured window mode:

```python
with EventReader("raw_file.raw", delta_t=10_000) as r:
    r.seek(t=2_000_000)   # skip to t = 2.0 s
    window = r.read()     # first delta_t window from there
    r.seek(n=1_000_000)   # or jump to the 1,000,000th event
```

Seeking uses an index (a Metavision `.tmp_index` sidecar when present, else one
built in memory) or exact record math, and falls back to iterate-and-skip on
non-seekable streams.

#### `dense`

Dense representations — turn a sparse event stream into fixed-size per-pixel
tensors: histograms, voxel grids, time surfaces, accumulation frames and TORE.
(A future `sparse` module will hold event graphs, sparse tensors and point
clouds.)

#### `filtering`

Selecting and dropping events: spatial masking today, with denoising, ROI and
downsampling to follow.

#### `transforms`

torchvision/tonic-style augmentation transforms (drops, spatial flips, jitter,
time skew/normalize, refractory filtering). Each composable `Transform` class
pairs with a pure `functional` kernel, and `Compose` chains them with minimal
unwrap/repack overhead.

#### `random`

Generating random events and adding noise to event recordings

#### `types`

The library is built around the `EventArray` type — a wrapper giving events a
**struct-of-arrays** (SoA) representation, which nearly every reader, writer and
transform exchanges:

- Fields (`Event_dtype`): `t` `int64` (signed 64-bit µs), `x`/`y` `uint16` (up to
  65,535 × 65,535 px), `p` `uint8`.
- **SoA** — four contiguous columns; cache-friendly, vectorizes over whole
  columns, no record padding: 8+2+2+1 = **13 bytes/event** (≈ 13 MB/MEv). Best
  for the column-wise processing that dominates event workloads.
- **Array-of-structs** equally supported (`from_aos` / `to_aos` / `np.asarray`,
  cheap); C-aligned record is **16 bytes/event**. Best for per-record iteration
  or an opaque buffer for another library/serializer. Transforms dispatch on
  input type, so you get back whichever form you passed in.
- Slicing, field subsetting, and an optional lightweight `metadata` dict (e.g.
  `sensor_size`) round out the type.


#### `vis`

The `vis` moduels provides several methods for visualizing the events (for example as histograms), but also provides a streamlined interface for more complex visualization techneques, such as using the [E2Vid](https://github.com/uzh-rpg/rpg_e2vid) reconstructor.


```python
from evutils.vis.reconstructor import RPG_Reconstructor

reconstructor = RPG_Reconstructor(1280, 720)

img = reconstructor.gen_frame(events)

```

## Running tests

Tests are managed via `pytest`. If you installed the package with the `[dev]` or `[test]` flag, you can run the standard test suite via:
```bash
uv run pytest -s
```

### Testing Docstrings
The library uses `doctest` to ensure all Python `>>>` examples inside docstrings are correct and functional. Because the default configuration only scans the `tests/` directory, you must explicitly tell pytest to scan the source code and ignore legacy submodules (like `rpg_e2vid` which contains Python 2 syntax):

```bash
uv run pytest --doctest-modules src/evutils --ignore=src/evutils/vis/reconstructor/rpg_e2vid/
```

## [Benchmarks](benchmarks/README.md)

In-RAM read/write **throughput** benchmarks live in `benchmarks/throughput.py` and are kept out of the normal test run. They report M events/s as two matrices (format × library). Run explicitly:

```bash
uv run python benchmarks/throughput.py                        # evutils + installed peers
uv run python benchmarks/throughput.py --dataset small --events 2_000_000   # quick smoke
```

The benchmark downloads a real Prophesee recording on first use, decodes a capped in-RAM payload, and measures every format on a RAM disk (`/dev/shm`). Optional cross-library comparisons (expelliarmus, evlib, evt3) light up automatically once installed (`uv pip install -e ".[compare]"`); OpenEB/Metavision is compared via the Docker image in `benchmarks/docker/`. See [`benchmarks/README.md`](benchmarks/README.md) for details.



## & Roadmap
We aim for universal event format support, prioritizing blazing fast read/write speeds, completeness, and extensibility. 
- [x] Universal format support (`.raw`, `.evt2`, `.dat`, `.aedat4`, `.hdf5`, `.npz`, `.csv`, etc.)
- [x] Full Read/Write parity where possible
- [x] Chunked & Streaming access
- [x] External trigger data parsing
- [x] **Random access / Timestamp indexing** (`EventReader.seek(t=/n=)` — by time or event index, forward/backward)
- [x] **Arbitrary input sources:** memory-mapped IO, pure in-memory streams (HTTP streams pending)
- [ ] **On-the-fly Compression wrappers:** passing file handles through `zstd` or `lz4` compression transparently before decoding
- [ ] **EventStreamer Pipeline Refactor:** Decouple `EventReader`'s monolithic chunking logic into composable functional generators in `chunking.py`, exposing a native `EventStreamer` for power-users while turning `EventReader` into a clean Façade. *(`EventStreamer` and generators complete; `EventReader` Facade pending).*


## Acknowledgements

Thanks to all the contributors for supporting this project:

* Elia Franc
* Jakub Mandula


## Cite
```bibtex
@PhDThesis{2024mandula_evutils,
  author        = {Jakub Mandula},
  title         = {EV-Utils: collection of utilities for working with event-based vision data},
  school        = {Dept. of Information Technology and Electrical Engineering, ETH Zurich},
  year          = 2024
}
```
