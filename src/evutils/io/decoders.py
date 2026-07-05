
import warnings
from pathlib import Path
from typing import Type

import numpy as np


from .common import EventDecoder

_READER_MAPPING = {}




from ._aedat import EventDecoder_Aedat
_READER_MAPPING[".aedat"] = EventDecoder_Aedat

from ._bin import EventDecoder_Bin
_READER_MAPPING[".bin"] = EventDecoder_Bin

from ._csv import EventDecoder_Csv
_READER_MAPPING[".csv"] = EventDecoder_Csv
_READER_MAPPING[".txt"] = EventDecoder_Csv


from ._dat import EventDecoder_Dat
_READER_MAPPING[".dat"] = EventDecoder_Dat

from ._hdf5 import EventDecoder_HDF5
_READER_MAPPING[".hdf5"] = EventDecoder_HDF5
_READER_MAPPING[".h5"] = EventDecoder_HDF5
from ._npz import EventDecoder_Npz
_READER_MAPPING[".npz"] = EventDecoder_Npz

from ._evt import EventDecoder_EVT

_READER_MAPPING[".raw"] = EventDecoder_EVT
_READER_MAPPING[".evt"] = EventDecoder_EVT
_READER_MAPPING[".evt3"] = EventDecoder_EVT
_READER_MAPPING[".evt2"] = EventDecoder_EVT
_READER_MAPPING[".evt21"] = EventDecoder_EVT

from ._aer import EventDecoder_AER
_READER_MAPPING[".aer"] = EventDecoder_AER


def get_reader_from_filename(file: Path) -> Type[EventDecoder]:
    '''
    Get the appropriate reader for the given file

    Parameters
    ----------
    file
        File to read

    Returns
    -------
    EventDecoder
        Reader object for the file
    '''


    ext = file.suffix.lower()
    if ext not in _READER_MAPPING:
        raise ValueError(f"File extension {ext} not supported, available extensions: {list(_READER_MAPPING.keys())}")

    reader_cls = _READER_MAPPING[ext]

    return reader_cls


# Content sniffers: (predicate over the first bytes -> decoder class). Tried in
# order when the filename extension is unknown or absent (streams, USB).
def _sniff_prophesee(head: bytes) -> bool:
    # Prophesee RAW/EVT files begin with an ASCII '% ...' header.
    return head[:1] == b"%"


_SNIFFERS = [
    (_sniff_prophesee, "EventDecoder_EVT"),
]


def resolve_decoder_cls(source) -> Type[EventDecoder]:
    '''
    Determine the decoder class for a :class:`ByteSource`.

    Tries the filename extension first (cheap, usually right), then falls back
    to sniffing the leading bytes -- which works for streams and USB devices
    that have no filename.

    Parameters
    ----------
    source
        A ByteSource (see :mod:`evutils.io._source`).

    Returns
    -------
    Type[EventDecoder]
        The decoder class to instantiate with the source.
    '''
    name = getattr(source, "name", None)
    if name:
        ext = Path(name).suffix.lower()
        if ext in _READER_MAPPING:
            return _READER_MAPPING[ext]

    try:
        head = source.peek(512)
    except Exception:
        head = b""

    for matches, cls_name in _SNIFFERS:
        if matches(head):
            return globals()[cls_name]

    raise ValueError(
        "Could not determine the event format: unknown/absent extension "
        f"({name!r}) and no known magic bytes. Pass an explicit decoder."
    )









__all__ = ["EventDecoder", 'EventDecoder_Aedat', 'EventDecoder_Bin', 'EventDecoder_Csv', 'EventDecoder_Dat', 'EventDecoder_HDF5', 'EventDecoder_Npz', 'EventDecoder_EVT', 'EventDecoder_AER', 'get_reader_from_filename', 'resolve_decoder_cls']
