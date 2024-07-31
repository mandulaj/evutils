
from ._reader import EventReader

import numpy as np


try:
    from ._bin import EventReader_Bin
except ImportError as e:
    error = str(e)
    class EventReader_Bin(EventReader):
        def __init__(self, file):
            raise ImportError("EventReader_Bin not available: " +  error)

try:
    from ._csv import EventReader_Csv
except ImportError as e:
    error = str(e)
    class EventReader_Csv(EventReader):
        def __init__(self, file):
            raise ImportError("EventReader_Csv not available: " +  error)
        
try:
    from ._dat import EventReader_Dat
except ImportError as e:
    error = str(e)
    class EventReader_Dat(EventReader):
        def __init__(self, file):
            raise ImportError("EventReader_Dat not available: " +  error)
        
try:
    from ._hdf5 import EventReader_HDF5
except ImportError as e:
    error = str(e)
    class EventReader_HDF5(EventReader):
        def __init__(self, file):
            raise ImportError("EventReader_HDF5 not available: " +  error)
        
try:
    from ._npz import EventReader_Npz
except ImportError as e:
    error = str(e)
    class EventReader_Npz(EventReader):
        def __init__(self, file):
            raise ImportError("EventReader_Npz not available: " +  error)

try:
    from ._raw import EventReader_RAW
except ImportError as e:
    error = str(e)
    class EventReader_RAW(EventReader):
        def __init__(self, file):
            raise ImportError("EventReader_RAW not available: " +  error)

try:
    from ._txt import EventReader_Txt
except ImportError as e:
    error = str(e)
    class EventReader_Txt(EventReader):
        def __init__(self, file):
            raise ImportError("EventReader_Txt not available: " +  error)
        



class EventReader_Any(EventReader):
    '''
    EventReader_Any: A class to automatically detect the format of the file and read it accordingly.
    '''

    def __init__(self, file):
        super().__init__(file)
        raise NotImplementedError("EventReader_Any is not implemented yet")


