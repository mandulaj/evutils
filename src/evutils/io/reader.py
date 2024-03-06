
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
        



# class EventReader_Any(EventReader):

#     def __init__(self, file):
#         super().__init__(file)



# class EventsIterator():
#     def __init__(self, reader, delta_t=30e3, mode='mixed', n_events=1e9, relative_timestamps=False):
#         self.reader = reader
#         self.delta_t = delta_t
#         self.mode = mode
#         self.n_events = n_events
#         self.relative_timestamps = relative_timestamps

#     def __iter__(self):
#         return self
    
#     def __next__(self):
#         pass