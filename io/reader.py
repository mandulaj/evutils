


from ._reader import EventReader



from ._bin import EventReader_Bin
from ._csv import EventReader_Csv
from ._dat import EventReader_Dat
from ._hdf5 import EventReader_HDF5
from ._npz import EventReader_Npz
from ._raw import EventReader_RAW
from ._txt import EventReader_Txt




class EventReader_Any(EventReader):

    def __init__(self, file):
        super().__init__(file)



class EventsIterator():
    def __init__(self, reader, delta_t=30e3, mode='mixed', n_events=1e9, relative_timestamps=False):
        self.reader = reader
        self.delta_t = delta_t
        self.mode = mode
        self.n_events = n_events
        self.relative_timestamps = relative_timestamps

    def __iter__(self):
        return self
    
    def __next__(self):
        pass