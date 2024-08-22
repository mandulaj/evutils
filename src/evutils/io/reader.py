
import numpy as np
import warnings

from ._common import EventFileReader

from pathlib import Path


_READER_MAPPING = {}


try:
    from ._aedat import EventFileReader_Aedat

    _READER_MAPPING[".aedat"] = EventFileReader_Aedat
except ImportError as e:
    error = str(e)
    warnings.warn("EventFileReader_Aedat not available: " +  error)

    class EventFileReader_Aedat(EventFileReader):
        def __init__(self, file):
            raise ImportError("EventFileReader_Aedat not available: " +  error)


try:
    from ._bin import EventFileReader_Bin

    _READER_MAPPING[".bin"] = EventFileReader_Bin
except ImportError as e:
    error = str(e)
    warnings.warn("EventFileReader_Bin not available: " +  error)

    class EventFileReader_Bin(EventFileReader):
        def __init__(self, file):
            raise ImportError("EventFileReader_Bin not available: " +  error)

try:
    from ._csv import EventFileReader_Csv

    _READER_MAPPING[".csv"] = EventFileReader_Csv
except ImportError as e:
    error = str(e)
    warnings.warn("EventFileReader_Csv not available: " +  error)

    class EventFileReader_Csv(EventFileReader):
        def __init__(self, file):
            raise ImportError("EventFileReader_Csv not available: " +  error)
        
try:
    from ._dat import EventFileReader_Dat

    _READER_MAPPING[".dat"] = EventFileReader_Dat
except ImportError as e:
    error = str(e)
    warnings.warn("EventFileReader_Dat not available: " +  error)

    class EventFileReader_Dat(EventFileReader):
        def __init__(self, file):
            raise ImportError("EventFileReader_Dat not available: " +  error)
        
try:
    from ._hdf5 import EventFileReader_HDF5

    _READER_MAPPING[".hdf5"] = EventFileReader_HDF5
    _READER_MAPPING[".h5"] = EventFileReader_HDF5
except ImportError as e:
    error = str(e)
    warnings.warn("EventFileReader_HDF5 not available: " +  error)

    class EventFileReader_HDF5(EventFileReader):
        def __init__(self, file):
            raise ImportError("EventFileReader_HDF5 not available: " +  error)
        
try:
    from ._npz import EventFileReader_Npz

    _READER_MAPPING[".npz"] = EventFileReader_Npz
except ImportError as e:
    error = str(e)
    warnings.warn("EventFileReader_Npz not available: " +  error)

    class EventFileReader_Npz(EventFileReader):
        def __init__(self, file):
            raise ImportError("EventFileReader_Npz not available: " +  error)

try:
    from ._raw import EventFileReader_RAW

    _READER_MAPPING[".raw"] = EventFileReader_RAW
except ImportError as e:
    error = str(e)
    warnings.warn("EventFileReader_RAW not available: " +  error)

    class EventFileReader_RAW(EventFileReader):
        def __init__(self, file):
            raise ImportError("EventFileReader_RAW not available: " +  error)

try:
    from ._txt import EventFileReader_Txt

    _READER_MAPPING[".txt"] = EventFileReader_Txt
except ImportError as e:
    error = str(e)
    warnings.warn("EventFileReader_Txt not available: " +  error)
    
    class EventFileReader_Txt(EventFileReader):
        def __init__(self, file):
            raise ImportError("EventFileReader_Txt not available: " +  error)



def get_file_reader(file: Path) -> EventFileReader:
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
    if ext not in _READER_MAPPING:
        raise ValueError(f"File extension {ext} not supported, available extensions: {list(_READER_MAPPING.keys())}")
    
    reader_cls = _READER_MAPPING[ext]

    return reader_cls









__all__ = ["EventFileReader", 'EventFileReader_Aedat', 'EventFileReader_Bin', 'EventFileReader_Csv', 'EventFileReader_Dat', 'EventFileReader_HDF5', 'EventFileReader_Npz', 'EventFileReader_RAW', 'EventFileReader_Txt', 'get_file_reader']