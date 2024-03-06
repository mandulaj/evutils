
from ._writer import EventWriter

import numpy as np

 
try:
    from ._bin import EventWriter_Bin
except ImportError as e:
    error = str(e)
    class EventWriter_Bin(EventWriter):
        def __init__(self, file, width=1280, height=720):
            raise ImportError("EventWriter_Bin not available: " +  error)
        
try:
    from ._csv import EventWriter_Csv
except ImportError as e:
    error = str(e)
    class EventWriter_Csv(EventWriter):
        def __init__(self, file, width=1280, height=720):
            raise ImportError("EventWriter_Csv not available: " +  error)

try:
    from ._dat import EventWriter_Dat
except ImportError as e:
    error = str(e)
    class EventWriter_Dat(EventWriter):
        def __init__(self, file, width=1280, height=720):
            raise ImportError("EventWriter_Dat not available: " +  error)

try:
    from  ._hdf5 import EventWriter_HDF5
except ImportError as e:
    error = str(e)
    class EventWriter_HDF5(EventWriter):
        def __init__(self, file, width=1280, height=720):
            raise ImportError("EventWriter_HDF5 not available: " +  error)

try:
    from ._npz import EventWriter_Npz
except ImportError as e:
    error = str(e)
    class EventWriter_Npz(EventWriter):
        def __init__(self, file, width=1280, height=720):
            raise ImportError("EventWriter_Npz not available: " +  error)

try:
    from ._raw import EventWriter_RAW
except ImportError as e:
    error = str(e)
    class EventWriter_RAW(EventWriter):
        def __init__(self, file, width=1280, height=720):
            raise ImportError("EventWriter_RAW not available: " +  error)

try:
    from ._txt import EventWriter_Txt
except ImportError as e:
    error = str(e)
    class EventWriter_Txt(EventWriter):
        def __init__(self, file, width=1280, height=720):
            raise ImportError("EventWriter_Txt not available: " +  error)





# class EventWriter_Any(EventWriter):

#     def __init__(self, file, width=1280, height=720):
#         super().__init__(file, width, height)



