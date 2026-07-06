# Supported Formats

`evutils.io` reads and writes event data through a single interface --
`EventReader` / `EventWriter` pick the right backend from the file extension
(or from magic bytes for extensionless streams). This page lists every format,
its support status, and format-specific options.

## Overview

| Format | Extensions | Read | Write | Backend | Notes |
|---|---|:---:|:---:|---|---|
| EVT3 (Prophesee RAW) | `.raw`, `.evt`, `.evt3` | ✅ | ✅ | C | external triggers; vectorized events |
| EVT2.1 (Prophesee RAW) | `.raw`, `.evt21` | ✅ | ✅ | C | write is one event per word (valid, not vectorized) |
| EVT2 (Prophesee RAW) | `.raw`, `.evt2` | ✅ | ✅ | C | |
| DAT (Prophesee) | `.dat` | ✅ | ✅ | C / numpy | 32-bit timestamp overflow tracked |
| AER (Prophesee) | `.aer` | ✅ | ✅ | C / numpy | no timestamps -- see [AER timestamps](#aer-timestamps) |
| AEDAT 1.0 / 2.0 / 3.1 / 4.0 | `.aedat`, `.aedat4` | ✅ | 🚧 planned | numpy | AEDAT4 compression needs `evutils[aedat]` |
| HDF5 (DSEC/RVT layout) | `.h5`, `.hdf5` | ✅ | ✅ | h5py | needs `evutils[hdf5]`; `ms_to_idx` random access |
| HDF5 (Prophesee layout) | `.h5`, `.hdf5` | ✅ | 🚧 planned | h5py | ECF-compressed files need the ECF codec plugin |
| NPZ | `.npz` | ✅ | ✅ | numpy | streaming both ways; `np.load`/`np.savez` compatible |
| CSV / TXT | `.csv`, `.txt` | ✅ | ✅ | pandas | needs `evutils[pandas]`; column order configurable |
| BIN | `.bin` | 🚧 planned | 🚧 planned | -- | reserved, raises `NotImplementedError` |

All decoders stream: only one chunk of events is held in memory at a time, so
arbitrarily large recordings can be iterated. `read_all()` is the explicit
opt-in that materialises a whole recording.

## Prophesee RAW / EVT

The `%`-header of a RAW file names the encoding (EVT2, EVT2.1 or EVT3); the
reader dispatches automatically. Decoding is done by the native C parsers,
including timestamp-overflow tracking (EVT3's 24-bit and EVT2/EVT2.1's 34-bit
rolling time bases are extended to 64-bit microsecond timestamps). External
trigger events are decoded when `EventReader(..., ext_trigger=True)`.

The writer selects the output encoding via `format=`:

```python
with EventWriter("out.raw", width=1280, height=720, format="evt3") as w:
    w.write(events)
```

## DAT

Prophesee's fixed-record CD-event format: an ASCII header, then 8 bytes per
event. Coordinates are 14-bit, timestamps 32-bit microseconds (the decoder
extends them past the ~71 minute wrap).

## AER

A raw 32-bit-per-event stream with 9-bit coordinates and **no header and no
timestamps** (GenX320-style). Coordinates above 511 cannot be represented.

(aer-timestamps)=
### AER timestamps

Because the format carries no time information, the decoder's `timestamps`
parameter selects how the `t` column is generated:

```python
EventReader("f.aer")                                      # t = 0 (default)
EventReader("f.aer", timestamps="sequential", t_step=10)  # t = 0, 10, 20, ...
EventReader("f.aer", timestamps=my_int64_array)           # user-provided
```

## AEDAT (jAER / cAER / DV)

All four AEDAT container versions are read; the version is detected from the
`#!AER-DATx.y` header line (headerless files default to 1.0, per jAER):

* **1.0** -- 6-byte big-endian records, DVS128 address layout.
* **2.0** -- 8-byte big-endian records. The address layout depends on the
  camera; select it with `EventReader(..., layout="davis")` (default, skips
  APS/IMU words) or `layout="dvs128"`.
* **3.1** -- cAER packet stream; polarity-event packets are decoded,
  frame/IMU/trigger packets are skipped.
* **4.0** -- DV-framework FlatBuffer packets. LZ4- or Zstd-compressed files
  need the optional dependencies: `pip install evutils[aedat]`.

Writing AEDAT is not implemented yet.

## HDF5

Requires `evutils[hdf5]` (h5py + hdf5plugin). Two on-disk layouts are read,
detected automatically:

* **DSEC / RVT layout** (also what the writer produces): the four columns
  under `events/{t,x,y,p}`, `width`/`height` file attributes, a `ms_to_idx`
  index and an optional DSEC `t_offset`. The index gives O(1) random access
  by time: `decoder.read(start_ms, end_ms)`.
* **Prophesee layout** (`.hdf5` from Metavision): a compound `CD/events`
  dataset. Prophesee compresses it with their ECF codec, which is a separate
  HDF5 plugin -- install it from
  [prophesee-ai/hdf5_ecf](https://github.com/prophesee-ai/hdf5_ecf) (or read
  the file with the plugin on `HDF5_PLUGIN_PATH`). Without the plugin,
  uncompressed Prophesee-layout files still work.

Writing uses the DSEC/RVT layout; writing the Prophesee layout is planned.

## NPZ

Events stored as four flat arrays `t`, `x`, `y`, `p` (plus `width`/`height`),
fully compatible with plain `np.savez` / `np.load`; a structured `events`
array is also accepted. Reading and writing stream chunk-by-chunk, so
recordings larger than memory are fine. `EventWriter(..., compressed=True)`
deflates the archive.

## CSV / TXT

Requires `evutils[pandas]`. A header line is auto-detected; the column order
is configurable on both ends (`order=['t', 'x', 'y', 'p']`), as are the
delimiter and (for writing) whether to emit a header.

## Testing status

Synthetic round-trip and edge-case tests cover every implemented format.
Real-camera recordings are currently only available for the EVT2 / EVT2.1 /
EVT3 tests (with ground-truth metadata); real-file fixtures for DAT, AER,
AEDAT (all versions), Prophesee HDF5 and DSEC HDF5 are still to be added.
