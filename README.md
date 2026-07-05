# <a href="https://mandulaj.github.io/evutils"><img src="https://mandulaj.github.io/evutils/_static/event_hexagon_broken.webp" alt="evutils_logo" width="50" align="top" style="background-color: #fff0;"></a> EV-Utils
[![PyPI Version](https://img.shields.io/pypi/v/evutils)](https://pypi.org/project/evutils/)
[![Test](https://github.com/mandulaj/evutils/actions/workflows/test.yaml/badge.svg)](https://github.com/mandulaj/evutils/actions/workflows/test.yaml)

## Overview
EV-Utils (``evutils``) is a performant collection of utilities for working with event-based vision data. Built with minimal dependencies, it relies on compiled C backed for speed while offering a clean, modular Python interface.



### Inspirations & Related Work 
This project draws inspiration from several excellent libraries in the event-based vision ecosystem and attempts to fill in their shortcomings:

* [Tonic](https://github.com/neuromorphs/tonic)
* [event_utils](https://github.com/TimoStoff/event_utils)
* [evlib](https://github.com/tallamjr/evlib)
* [expelliarmus](https://github.com/open-neuromorphic/expelliarmus)
* [openeb](https://github.com/prophesee-ai/openeb)


## Installation
We recommend installing `evutils` using `uv`. 
### From PyPi
```bash
uv add evutils # Basic library
uv add evutils[all] # All groups (pandas, numba, torch, hdf5, etc..)
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
└── dataset     - Wrappers for various dataset loaders
└── events      - Core event handling logic
└── io          - Event reading and writing interfaces
    ├── reader 
    └── writer
└── random      - Random event generation and noise injection
└── torch       - PyTorch integration (requires evutils[torch])
└── types       - Standard types for representing Events in NumPy arrays
└── vis         - Visualization methods
    ├── histogram
    └── reconstructor
```

### Quick API overview 

<!-- ### `augment`

Event augmentations

### `dataset`

Wrappers for various dataset loaders -->

#### `io`: Reading and Writing Events

The `io` module provides methods for reading and writing events into various event formats. It provides a simple `.read()` and `.write()` interface as well as more advanced interfaces using iterators and slicing.

```python
from evutils.io import EventReader


ev_file = EventReader("raw_file.raw", delta_t=10e3)

events = ev_file.read()

```

#### `utils`

Various utility functions

#### `random`

Generating random events and adding noise to event recordings

#### `types`

This provides several standard types for representing Events in numpy arrays


#### `vis`

The `vis` moduels provides several methods for visualizing the events (for example as histograms), but also provides a streamlined interface for more complex visualization techneques, such as using the [E2Vid](https://github.com/uzh-rpg/rpg_e2vid) reconstructor.


```python
from evutils.vis.reconstructor import RPG_Reconstructor

reconstructor = RPG_Reconstructor(1280, 720)

img = reconstructor.gen_frame(events)

```

## Running tests

Tests are managed via `pytest`. If you installed the package with the `[dev]` or `[test]` flag, you can run them via:
```bash
uv run pytest -s
```


## [Benchmarks](benchmarks/README.md)

Read/write throughput benchmarks (using [pytest-benchmark](https://pytest-benchmark.readthedocs.io)) live in `benchmarks/` and are kept out of the normal test run. Run them explicitly:

```bash
uv run pytest benchmarks/                                   # evutils only
uv run pytest benchmarks/ --benchmark-group-by=param:fmt    # compare libraries per format
```

The benchmarks download a real Prophesee recording on first use. Optional cross-library comparisons run automatically once the libraries are installed (`uv pip install -e ".[compare]"`); OpenEB/Metavision is compared via the Docker image in `benchmarks/docker/`. See [`benchmarks/README.md`](benchmarks/README.md) for details.


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