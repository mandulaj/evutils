# Supported Formats

`evutils.io` reads and writes event data through a single interface --
`EventReader` / `EventWriter` pick the right backend from the file extension
(or from magic bytes for extensionless streams). This page lists every format,
its support status, and format-specific options.

## The `EventArray` representation

Everything the readers and writers exchange is an `EventArray`
(`evutils.types`) -- a lightweight wrapper storing the four event fields as
separate parallel NumPy arrays, a **struct-of-arrays** (SoA) layout.

Fields (`Event_dtype`):

- `t` -- `int64`, signed 64-bit microsecond timestamp (µs range ~±292,000 years).
- `x`, `y` -- `uint16`, up to 65,535 × 65,535 pixel resolution.
- `p` -- `uint8`, polarity.

Why SoA:

- Native layout of the C parsers.
- Column access is contiguous and cache-friendly; NumPy and the Numba-JIT'd
  transforms vectorize over whole columns.
- No per-record alignment padding: 8+2+2+1 = **13 bytes/event** (≈ 13 MB per
  million events).

### Array-of-structs is supported too

The array-of-structs (AoS) form -- one NumPy structured array of `(t, x, y, p)`
records (`dtype = Event_dtype`) -- is first-class. Conversions are cheap:

```python
import numpy as np
from evutils.types import EventArray, Event_dtype

aos  = np.zeros(1000, dtype=Event_dtype)  # array-of-structs
soa  = EventArray.from_aos(aos)           # -> struct-of-arrays
back = soa.to_aos()                       # -> array-of-structs
same = np.asarray(soa)                    # __array__ also yields AoS
```

- Transforms/helpers **dispatch on input type**: ndarray in gives ndarray out,
  `EventArray` in gives `EventArray` out -- stay in whichever form your code uses.
- **AoS** for per-record iteration, a single opaque buffer for another library or
  serializer (`np.savez`, memory-mapping, packed-record C code), or sorting whole
  events. C-aligned record is 8+2+2+1 + 3 pad = **16 bytes/event**.
- **SoA** for the column-wise vectorized processing that dominates event
  workloads (per-field math, masking, JIT kernels). `evutils` keeps SoA as the
  working form and drops to AoS at the boundaries.

An optional lightweight `metadata` dict (e.g. `sensor_size`) rides along on the
`EventArray`, populated by readers and consumed by writers where the format
carries geometry.

## IO Roadmap & Goals

We are aiming for universal event format support with the highest possible performance and extensibility:

- [x] Full Read/Write parity where possible
- [x] Chunked & Streaming access
- [x] External trigger data decoding
- [ ] Random access / Indexing (`ms_to_idx` implemented for HDF5, big TODO for `.raw` streams)
- [ ] Arbitrary inputs (file-like objects, `io.BytesIO`, memory-mapped files)
- [ ] Compression wrappers (e.g., passing streams through `zstd` transparently)

## Format Matrix

| Format                      | Extensions              |   Read    |   Write   | Backend   | Notes                                                      |
| --------------------------- | ----------------------- | :-------: | :-------: | --------- | ---------------------------------------------------------- |
| EVT4 (Prophesee RAW)        | `.raw`, `.evt`, `.evt4` |     🚧     |     🚧     | C         | external triggers; vectorized events                       |
| EVT3 (Prophesee RAW)        | `.raw`, `.evt`, `.evt3` |     ✅     |     ✅     | C         | external triggers; vectorized events                       |
| EVT2.1 (Prophesee RAW)      | `.raw`, `.evt21`        |     ✅     |     ✅     | C         | write is one event per word (valid, not vectorized)        |
| EVT2 (Prophesee RAW)        | `.raw`, `.evt2`         |     ✅     |     ✅     | C         |                                                            |
| DAT (Prophesee)             | `.dat`                  |     ✅     |     ✅     | C / numpy | 32-bit timestamp overflow tracked                          |
| AER (Prophesee)             | `.aer`                  |     ✅     |     ✅     | C / numpy | no timestamps -- see [AER timestamps](#aer-timestamps)     |
| AEDAT 1.0 / 2.0 / 3.1 / 4.0 | `.aedat`, `.aedat4`     |     ✅     | 🚧 planned | numpy     | AEDAT4 compression needs `evutils[aedat]`                  |
| HDF5 (DSEC/RVT layout)      | `.h5`, `.hdf5`          |     ✅     |     ✅     | h5py      | needs `evutils[hdf5]`; `ms_to_idx` random access           |
| HDF5 (Prophesee layout)     | `.h5`, `.hdf5`          |     ✅     | 🚧 planned | h5py      | ECF-compressed files need the ECF codec plugin             |
| NPZ                         | `.npz`                  |     ✅     |     ✅     | numpy     | streaming both ways; `np.load`/`np.savez` compatible       |
| CSV / TXT                   | `.csv`, `.txt`          |     ✅     |     ✅     | C         | native C parser (no extra deps); column order configurable |
| BIN                         | `.bin`                  | 🚧 planned | 🚧 planned | --        | reserved, raises `NotImplementedError`                     |

All decoders stream: only one chunk of events is held in memory at a time, so
arbitrarily large recordings can be iterated. `read_all()` is the explicit
opt-in that materialises a whole recording.

## Format overview

# Prophesee Event Format Comparison

A side-by-side reference for Prophesee's event stream encodings (AER, EVT2.0, EVT2.1, EVT3.0, EVT4)
and the decoded file formats (DAT, CSV).

| Property                | AER                                                                     | EVT2.0                                                                   | EVT2.1                                                                    | EVT3.0                                                                   | EVT4                       | DAT                                                                 | CSV                                                                                     |
| ----------------------- | ----------------------------------------------------------------------- | ------------------------------------------------------------------------ | ------------------------------------------------------------------------- | ------------------------------------------------------------------------ | -------------------------- | ------------------------------------------------------------------- | --------------------------------------------------------------------------------------- |
| **Word size**           | 19-bit address                                                          | 32-bit                                                                   | 64-bit                                                                    | 16-bit                                                                   | 32-bit                     | 64-bit / event (8 B)                                                | text line                                                                               |
| **Vectorized**          | no                                                                      | no                                                                       | yes (32 px / word)                                                        | yes (12+12+8)                                                            | yes (32 px / 2 words)      | n/a¹                                                                | n/a¹                                                                                    |
| **Stateful decode**     | no                                                                      | minimal                                                                  | minimal                                                                   | heavy (differential)                                                     | minimal (+ vector pairing) | no                                                                  | no                                                                                      |
| **Timestamp**           | implicit (none stored)                                                  | 34-bit (6 low + 28 high)                                                 | 34-bit (6 + 28)                                                           | 24-bit (12 + 12) + loop                                                  | 34-bit (6 + 28)            | absolute 32-bit µs                                                  | absolute µs (text)                                                                      |
| **Bytes / single CD**   | ~2.4 (19-bit)                                                           | 4                                                                        | 8                                                                         | 2                                                                        | 4                          | 8                                                                   | ~15–20 (ASCII)                                                                          |
| **Bytes / 32-px burst** | ~76                                                                     | 128                                                                      | 8                                                                         | ~6–8                                                                     | 8                          | 256                                                                 | ~500+                                                                                   |
| **Best for**            | neuromorphic HW                                                         | low rate, robust                                                         | high rate                                                                 | highest rate / most compact                                              | EVT2 datapath + vectors    | offline processing / legacy tooling                                 | human inspection / interchange                                                          |
| **Docs**                | [link](https://docs.prophesee.ai/stable/data/encoding_formats/aer.html) | [link](https://docs.prophesee.ai/stable/data/encoding_formats/evt2.html) | [link](https://docs.prophesee.ai/stable/data/encoding_formats/evt21.html) | [link](https://docs.prophesee.ai/stable/data/encoding_formats/evt3.html) | none (HAL only)            | [link](https://docs.prophesee.ai/stable/data/file_formats/dat.html) | [File-to-CSV](https://docs.prophesee.ai/stable/samples/modules/stream/file_to_csv.html) |

## Caveats

The columns are not strictly apples-to-apples:

1. **DAT and CSV are not sensor-side encodings** — they store *already-decoded* events, so
   "vectorized" and "stateful decode" don't really apply. A DAT CD event is a fixed 8-byte (64-bit)
   word: a `uint32` timestamp plus a data word (event size is 8 for all common types; EventCd = `0x0C`,
   EventExtTrigger = `0x0E`). Timestamps are stored absolutely per event, so there's no high/low split
   to track. CSV isn't a native recorded format — it's produced by the File-to-CSV sample, which
   converts a DAT file to text (one event per line, e.g. `x, y, polarity, t`).
2. **EVT4 has no public spec** — that entire column is reverse-engineered from the Metavision HAL
   headers, so treat the numbers as inference, not documented fact. EVT4 is absent from Prophesee's
   official format list, which covers only EVT2.0 / 2.1 / 3.0.
3. **The 32-px burst row is best-case** — it assumes 32 same-row, same-polarity, same-timestamp
   events, which is exactly what the vectorized formats are optimized for and where
   EVT2.0 / DAT / CSV look worst.

## Sources

- Data Encoding Formats index: <https://docs.prophesee.ai/stable/data/encoding_formats/index.html>
- Recorded File Formats index: <https://docs.prophesee.ai/stable/data/file_formats/index.html>
- EVT4 details are derived from the Metavision HAL `evt4_*` headers (no public documentation).

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

- **1.0** -- 6-byte big-endian records, DVS128 address layout.
- **2.0** -- 8-byte big-endian records. The address layout depends on the
  camera; select it with `EventReader(..., layout="davis")` (default, skips
  APS/IMU words) or `layout="dvs128"`.
- **3.1** -- cAER packet stream; polarity-event packets are decoded,
  frame/IMU/trigger packets are skipped.
- **4.0** -- DV-framework FlatBuffer packets. LZ4- or Zstd-compressed files
  need the optional dependencies: `pip install evutils[aedat]`.

Writing AEDAT is not implemented yet.

## HDF5

Requires `evutils[hdf5]` (h5py + hdf5plugin). Two on-disk layouts are read,
detected automatically:

- **DSEC / RVT layout** (also what the writer produces): the four columns
  under `events/{t,x,y,p}`, `width`/`height` file attributes, a `ms_to_idx`
  index and an optional DSEC `t_offset`. The index gives O(1) random access
  by time: `decoder.read(start_ms, end_ms)`.
- **Prophesee layout** (`.hdf5` from Metavision): a compound `CD/events`
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

Parsed and written by the native C backend -- no extra dependencies.
A header line is auto-detected; the column order
is configurable on both ends (`order=['t', 'x', 'y', 'p']`), as are the
delimiter and (for writing) whether to emit a header.

## Testing status

Synthetic round-trip and edge-case tests cover every implemented format.
Real-camera recordings are currently only available for the EVT2 / EVT2.1 /
EVT3 tests (with ground-truth metadata); real-file fixtures for DAT, AER,
AEDAT (all versions), Prophesee HDF5 and DSEC HDF5 are still to be added.
