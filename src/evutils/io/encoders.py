"""Encoders module.

Provides mapping and retrieval of event encoders based on file extensions.
Backends whose optional dependencies are missing (e.g. pandas for CSV, h5py
for HDF5) are not registered; asking for them raises an ``ImportError`` that
names the extra to install.
"""

from pathlib import Path
from typing import Type

from .common import EventEncoder

#: Extension -> encoder class, for available backends only.
_WRITER_MAPPING: dict[str, Type[EventEncoder]] = {}

#: Extension -> reason it is unavailable (missing optional dependency).
_UNAVAILABLE: dict[str, str] = {}


from ._aedat import EventEncoder_Aedat
_WRITER_MAPPING[".aedat"] = EventEncoder_Aedat

from ._bin import EventEncoder_Bin
_WRITER_MAPPING[".bin"] = EventEncoder_Bin

try:
    from ._csv import EventEncoder_Csv
    _WRITER_MAPPING[".csv"] = EventEncoder_Csv
    _WRITER_MAPPING[".txt"] = EventEncoder_Csv
except ImportError:
    _UNAVAILABLE[".csv"] = _UNAVAILABLE[".txt"] = (
        "writing CSV/TXT event files requires pandas: install `evutils[pandas]`"
    )

from ._dat import EventEncoder_Dat
_WRITER_MAPPING[".dat"] = EventEncoder_Dat

try:
    from ._hdf5 import EventEncoder_HDF5
    _WRITER_MAPPING[".h5"] = EventEncoder_HDF5
    _WRITER_MAPPING[".hdf5"] = EventEncoder_HDF5
except ImportError:
    _UNAVAILABLE[".h5"] = _UNAVAILABLE[".hdf5"] = (
        "writing HDF5 event files requires h5py/hdf5plugin: install `evutils[hdf5]`"
    )

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
    if ext in _WRITER_MAPPING:
        return _WRITER_MAPPING[ext]
    if ext in _UNAVAILABLE:
        raise ImportError(f"File extension {ext} is supported, but {_UNAVAILABLE[ext]}")
    raise ValueError(
        f"File extension {ext} not supported, available extensions: "
        f"{sorted(_WRITER_MAPPING.keys() | _UNAVAILABLE.keys())}"
    )


__all__ = ["EventEncoder", "EventEncoder_Aedat", "EventEncoder_Bin", "EventEncoder_Csv", "EventEncoder_Dat", "EventEncoder_HDF5", "EventEncoder_Npz", "EventEncoder_EVT", "EventEncoder_AER", "get_file_writer"]
