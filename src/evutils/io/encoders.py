"""Encoders module.

Provides mapping and retrieval of event encoders based on file extensions.
"""

import warnings
from pathlib import Path
from typing import Type

import numpy as np

from .common import EventEncoder

_WRITER_MAPPING: dict[str, Type[EventEncoder]] = {}
from ._aedat import EventEncoder_Aedat
_WRITER_MAPPING[".aedat"] = EventEncoder_Aedat

from ._bin import EventEncoder_Bin
_WRITER_MAPPING[".bin"] = EventEncoder_Bin

from ._csv import EventEncoder_Csv
_WRITER_MAPPING[".csv"] = EventEncoder_Csv
_WRITER_MAPPING[".txt"] = EventEncoder_Csv

from ._dat import EventEncoder_Dat
_WRITER_MAPPING[".dat"] = EventEncoder_Dat

from ._hdf5 import EventEncoder_HDF5
_WRITER_MAPPING[".h5"] = EventEncoder_HDF5
_WRITER_MAPPING[".hdf5"] = EventEncoder_HDF5

from ._npz import EventEncoder_Npz
_WRITER_MAPPING[".npz"] = EventEncoder_Npz

from ._evt import EventEncoder_EVT
_WRITER_MAPPING[".raw"] = EventEncoder_EVT
_WRITER_MAPPING[".evt"] = EventEncoder_EVT
_WRITER_MAPPING[".evt3"] = EventEncoder_EVT

from ._aer import EventEncoder_AER
_WRITER_MAPPING[".aer"] = EventEncoder_AER



def get_file_writer(file: Path) -> Type[EventEncoder]:
    """Get the appropriate writer for the given file.

    Parameters
    ----------
    file
        File to write

    Returns
    -------
    EventFileWriter
        Writer object for the file

    """
    ext = file.suffix.lower()
    if ext not in _WRITER_MAPPING:
        raise ValueError(f"File extension {ext} not supported, available extensions: {list(_WRITER_MAPPING.keys())}")

    reader_cls = _WRITER_MAPPING[ext]

    return reader_cls




__all__ = ["EventEncoder", "EventEncoder_Aedat", "EventEncoder_Bin", "EventEncoder_Csv", "EventEncoder_Dat", "EventEncoder_HDF5", "EventEncoder_Npz", "EventEncoder_EVT", "EventEncoder_AER", "get_file_writer"]
