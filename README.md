EV Utils
========

EV-utils is a collection of utilities for working event based data inspired by the [event_utils](https://github.com/TimoStoff/event_utils) library. This library aims at being camera independent (yet also supporting specific camera vendors) with minimal dependencies but also performent. The library is divided into severla modules some of which can be used without installing all the dependencies. These include:

```
└── io
    ├── reader 
    └── writer
└── types
└── vis
    ├── histogram
    └── reconstructor
```


## Installation

### From Git
```
git clone --recurse-submodules git@git.ee.ethz.ch:pbl/research/event-camera/evutils.git

cd evutils
pip install . 
pip install -e . # Use this to install an editable version of the package
```

## Usage 

### `io`

The `io` module provides methods for reading and writing events into various event formats. It provides a simple `.read()` and `.write()` interface as well as more advanced interfaces using iterators and slicing.

```python
from evutils.io.reader import EventReader_RAW


ev_file = EventReader_RAW("raw_file.raw", delta_t=10e3)

events = ev_file.read()

```


### `types`

This provides several standard types for representing Events in numpy arrays


### `vis`

The `vis` moduels provides several methods for visualizing the events (for example as histograms), but also provides a streamlined interface for more complex visualization techneques, such as using the [E2Vid](https://github.com/uzh-rpg/rpg_e2vid) reconstructor.


```python
from evutils.vis.reconstructor import RPG_Reconstructor

reconstructor = RPG_Reconstructor(1280, 720)

img = reconstructor.gen_frame(events)

```


## Acknowledgements

Thanks to all the contributors for supporting this project:

* Elia Franc
* Jakub Mandula
