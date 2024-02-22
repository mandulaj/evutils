
from ._writer import EventWriter
from ._reader import EventReader

import numpy as np
import numba as nb

import h5py
import hdf5plugin



class EventWriter_HDF5(EventWriter):
    def __init__(self, file, width=1280, height=720, buffersize=10000):
        super().__init__(file, width, height)

        self.fd = h5py.File(self.file, "w")
        self.events = self.fd.create_group("events")
        self.compressor = hdf5plugin.Blosc(cname="zstd", clevel=5, shuffle=hdf5plugin.Blosc.SHUFFLE)
        self.buffer = np.empty(0, dtype=[("x", "uint16"), ("y", "uint16"), ("p", "uint8"), ("t", "uint32")])
        self.buffersize = buffersize
        self.n_buffers = 0
        self.ms_to_idx = [0]
        self.initialized = False

    def write(self, events: np.ndarray):
        self.buffer = np.append(self.buffer, events)
        self._set_ms_idx_for_events(self.buffer)
        if len(self.buffer) >= self.buffersize:
            n_full_buffers = len(self.buffer) // self.buffersize
            for i in range(n_full_buffers):
                buffer = self.buffer[i * self.buffersize : (i + 1) * self.buffersize]
                if not self.initialized:
                    self._initial_dataset_creation(buffer)
                    self.initialized = True
                else:
                    self._append_new_events(buffer)
                self.n_buffers += 1
            self.buffer = self.buffer[n_full_buffers * self.buffersize :]

    def close(self):
        self._append_new_events(self.buffer)
        self.ms_to_idx.append(self.x.shape[0])
        self.fd.create_dataset("ms_to_idx", data=self.ms_to_idx, **self.compressor)
        self.fd.close()

    def _set_ms_idx_for_events(self, events: np.ndarray):
        events_ms = events["t"] // 1000
        unique_ms = np.unique(events_ms)
        for ms in unique_ms:
            if ms <= len(self.ms_to_idx) - 1:
                continue
            self.ms_to_idx.append(self.n_buffers * self.buffersize + np.where(events_ms == ms)[0].min())

    def _initial_dataset_creation(self, event_buffer: np.ndarray):
        assert len(event_buffer) == self.buffersize, "Events must have the length of the buffer size."
        self.x = self.events.create_dataset(
            "x", data=event_buffer["x"], chunks=(self.buffersize,), maxshape=(None,), dtype="uint16", **self.compressor
        )
        self.y = self.events.create_dataset(
            "y", data=event_buffer["y"], chunks=(self.buffersize,), maxshape=(None,), dtype="uint16", **self.compressor
        )
        self.p = self.events.create_dataset(
            "p", data=event_buffer["p"], chunks=(self.buffersize,), maxshape=(None,), dtype="uint8", **self.compressor
        )
        self.t = self.events.create_dataset(
            "t", data=event_buffer["t"], chunks=(self.buffersize,), dtype="uint32", maxshape=(None,), **self.compressor
        )
        return self.x, self.y, self.p, self.t

    def _append_new_events(self, events: np.ndarray):
        n_events = events.shape[0]
        self.x.resize((self.x.shape[0] + n_events), axis=0)
        self.x[-n_events:] = events["x"]
        self.y.resize((self.y.shape[0] + n_events), axis=0)
        self.y[-n_events:] = events["y"]
        self.p.resize((self.p.shape[0] + n_events), axis=0)
        self.p[-n_events:] = events["p"]
        self.t.resize((self.t.shape[0] + n_events), axis=0)
        self.t[-n_events:] = events["t"]

class EventWriter_HDF5(EventWriter):
    def __init__(self, file, width=1280, height=720):
        super().__init__(file, width, height)