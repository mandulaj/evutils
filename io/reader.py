


class EventReader():
    def __init__(self):
        pass


class EventReader_RAW(EventReader):
    def __init__(self, file):
        super().__init__(file)


class EventReader_CSV(EventReader):
    def __init__(self, file):
        super().__init__(file)

class EventReader_HDF5(EventReader):
    def __init__(self, file):
        super().__init__(file)

class EventReader_Bin(EventReader):
    def __init__(self, file):
        super().__init__(file)


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