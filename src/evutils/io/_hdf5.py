import h5py
import hdf5plugin
import numba as nb
import numpy as np

from ..types import Events
from ._reader import EventReader
from ._writer import EventWriter



@nb.njit
def get_idx(events, ms_to_idx, last_ms_idx, n_written_events, max_ms, offset):
    idx = 0
    for ms in range(last_ms_idx, max_ms+1):
        while idx < len(events) and events['t'][idx] // 1000 < ms:
            idx += 1

        ms_to_idx[ms] = max(idx + offset + n_written_events, 0)


class EventWriter_HDF5(EventWriter):
    def __init__(self, file, width=1280, height=720, chunksize=10000):
        super().__init__(file, width, height)

        self.chunksize = chunksize

        self.ms_to_idx = np.empty(0, dtype=np.uint64)
        self.last_ms_idx = 0



    def init(self):
        if self.is_initialized:
            return

        self.fd = h5py.File(self.file, "w")
        self.events_group_h5 = self.fd.create_group("events")
        self.compressor = hdf5plugin.Blosc(cname="zstd", clevel=5, shuffle=hdf5plugin.Blosc.SHUFFLE)

        self.fd.attrs['width'] = self.width
        self.fd.attrs['height'] = self.height

        # Create datasets
        self.events_group_h5.create_dataset("x", shape=(0,), chunks=(self.chunksize, ), maxshape=(None,), 
                                            dtype="uint16", **self.compressor)
        self.events_group_h5.create_dataset("y", shape=(0,), chunks=(self.chunksize, ), maxshape=(None,),
                                             dtype="uint16", **self.compressor)
        self.events_group_h5.create_dataset("p", shape=(0,), chunks=(self.chunksize, ), maxshape=(None,),
                                             dtype="uint8", **self.compressor)
        self.events_group_h5.create_dataset("t", shape=(0,), chunks=(self.chunksize, ), maxshape=(None,),
                                             dtype="uint32", **self.compressor)
    

        self.is_initialized = True

    def write(self, events: np.ndarray):
        if not self.is_initialized:
            self.init()

        # Generate ms_to_idx
        self.__get_ms_idx_for_events(events)
            
        # Append events
        self.__append_new_events(events)



    def close(self):
        if not self.is_initialized:
            return
        
        # Write the ms_to_idx
        self.fd.create_dataset("ms_to_idx", data=self.ms_to_idx, dtype="uint64", **self.compressor)

        self.fd['ms_to_idx'].resize((len(self.ms_to_idx),))
        self.fd['ms_to_idx'][:] = self.ms_to_idx

        self.fd.close()
    
    def __get_ms_idx_for_events(self, events: np.ndarray, offset=-1):
        max_ms = int(events["t"][-1] // 1000)

        if max_ms + 1 > len(self.ms_to_idx):
            self.ms_to_idx.resize(max_ms + 1, refcheck=False)

        get_idx(events, self.ms_to_idx, self.last_ms_idx, self.n_written_events, max_ms, offset)


        self.last_ms_idx = max_ms + 1





    def __append_new_events(self, events: np.ndarray):
        n_events = events.shape[0]
        x = self.events_group_h5["x"]
        y = self.events_group_h5["y"]
        p = self.events_group_h5["p"]
        t = self.events_group_h5["t"]

        x.resize((x.shape[0] + n_events), axis=0)
        y.resize((y.shape[0] + n_events), axis=0)
        p.resize((p.shape[0] + n_events), axis=0)
        t.resize((t.shape[0] + n_events), axis=0)

        x[-n_events:] = events["x"]
        y[-n_events:] = events["y"]
        p[-n_events:] = events["p"]
        t[-n_events:] = events["t"]

        self.n_written_events += n_events



class EventReader_HDF5(EventReader):
    def __init__(self, file, width=1280, height=720):
        super().__init__(file, width, height)

    def init(self):
        if self.is_initialized:
            return
        self.fd = h5py.File(self.file, "r")
        self.ms_to_idx = np.asarray(self.fd["ms_to_idx"])
        self.max_events = self.fd["events"]["x"].shape[0]
        self.last_ms = len(self.ms_to_idx) - 1
        self.is_initialized = True

    def read(self, start_ms: int = 0, end_ms: int = -1) -> np.ndarray:
        if not self.is_initialized:
            self.init()

        if end_ms == -1:
            end_ms = self.last_ms

        assert start_ms >= 0, "start_ms must be greater or equal to 0"
        assert start_ms < end_ms, "start_ms must be smaller than end_ms"
        assert end_ms <= self.last_ms, f"end_ms must be smaller or equal to the last ms ({self.last_ms})"

        start_idx = self.ms_to_idx[start_ms]
        end_idx = self.ms_to_idx[end_ms]
        return self._read_events(start_idx, end_idx)

    def _read_events(self, start_idx: int, end_idx: int) -> np.ndarray:
        x = self.fd["events"]["x"][start_idx:end_idx]
        y = self.fd["events"]["y"][start_idx:end_idx]
        p = self.fd["events"]["p"][start_idx:end_idx]
        t = self.fd["events"]["t"][start_idx:end_idx]
        return np.array(list(zip(t, x, y, p)), dtype=Events)
