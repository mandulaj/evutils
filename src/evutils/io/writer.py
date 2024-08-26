
import numpy as np
import warnings

from ._common import  EventFileWriter

from pathlib import Path


_WRITER_MAPPING = {}

try:
    from ._aedat import EventFileWriter_Aedat
    _WRITER_MAPPING[".aedat"] = EventFileWriter_Aedat
except ImportError as e:
    error = str(e)
    warnings.warn("EventFileWriter_Aedat not available: " +  error)

    class EventFileWriter_Aedat(EventFileWriter):
        def __init__(self, file):
            raise ImportError("EventFileWriter_Aedat not available: " +  error)


try:
    from ._bin import EventFileWriter_Bin
    _WRITER_MAPPING[".bin"] = EventFileWriter_Bin
except ImportError as e:
    error = str(e)
    warnings.warn("EventFileWriter_Bin not available: " +  error)

    class EventFileWriter_Bin(EventFileWriter):
        def __init__(self, file, width=1280, height=720):
            raise ImportError("EventFileWriter_Bin not available: " +  error)
        
try:
    from ._csv import EventFileWriter_Csv
    _WRITER_MAPPING[".csv"] = EventFileWriter_Csv
except ImportError as e:
    error = str(e)
    warnings.warn("EventFileWriter_Csv not available: " +  error)
    
    class EventFileWriter_Csv(EventFileWriter):
        def __init__(self, file, width=1280, height=720):
            raise ImportError("EventFileWriter_Csv not available: " +  error)

try:
    from ._dat import EventFileWriter_Dat
    _WRITER_MAPPING[".dat"] = EventFileWriter_Dat
except ImportError as e:
    error = str(e)
    warnings.warn("EventFileWriter_Dat not available: " +  error)

    class EventFileWriter_Dat(EventFileWriter):
        def __init__(self, file, width=1280, height=720):
            raise ImportError("EventFileWriter_Dat not available: " +  error)

try:
    from ._hdf5 import EventFileWriter_HDF5
    _WRITER_MAPPING[".h5"] = EventFileWriter_HDF5
    _WRITER_MAPPING[".hdf5"] = EventFileWriter_HDF5
except ImportError as e:
    error = str(e)
    warnings.warn("EventFileWriter_HDF5 not available: " +  error)

    class EventFileWriter_HDF5(EventFileWriter):
        def __init__(self, file, width=1280, height=720):
            raise ImportError("EventFileWriter_HDF5 not available: " +  error)

try:
    from ._npz import EventFileWriter_Npz
    _WRITER_MAPPING[".npz"] = EventFileWriter_Npz
except ImportError as e:
    error = str(e)
    warnings.warn("EventFileWriter_Npz not available: " +  error)

    class EventFileWriter_Npz(EventFileWriter):
        def __init__(self, file, width=1280, height=720):
            raise ImportError("EventFileWriter_Npz not available: " +  error)

try:
    from ._raw import EventFileWriter_RAW
    _WRITER_MAPPING[".raw"] = EventFileWriter_RAW
except ImportError as e:
    error = str(e)
    warnings.warn("EventFileWriter_RAW not available: " +  error)

    class EventFileWriter_RAW(EventFileWriter):
        def __init__(self, file, width=1280, height=720):
            raise ImportError("EventFileWriter_RAW not available: " +  error)

try:
    from ._txt import EventFileWriter_Txt
    _WRITER_MAPPING[".txt"] = EventFileWriter_Txt
except ImportError as e:
    warnings.warn("EventFileWriter_Txt not available: " +  error)

    error = str(e)
    class EventFileWriter_Txt(EventFileWriter):
        def __init__(self, file, width=1280, height=720):
            raise ImportError("EventFileWriter_Txt not available: " +  error)

def get_file_writer(file: Path) -> EventFileWriter:
    '''
    Get the appropriate reader for the given file

    Parameters
    ----------
    file
        File to read

    Returns
    -------
    EventFileReader
        Reader object for the file
    '''


    ext = file.suffix.lower()
    if ext not in _WRITER_MAPPING:
        raise ValueError(f"File extension {ext} not supported, available extensions: {list(_WRITER_MAPPING.keys())}")
    
    reader_cls = _WRITER_MAPPING[ext]

    return reader_cls




__all__ = ["EventFileWriter", "EventFileWriter_Aedat", "EventFileWriter_Bin", "EventFileWriter_Csv", "EventFileWriter_Dat", "EventFileWriter_HDF5", "EventFileWriter_Npz", "EventFileWriter_RAW", "EventFileWriter_Txt", "get_file_writer"]
