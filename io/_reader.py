
import numpy as np

import os

class EventReader():
    READING_MODES = ["delta_t", "n_events", "mixed", "all", "auto"]
    def __init__(self, file, delta_t=None, n_events=None, max_events=10000000, mode="auto"):


        self.file = file
        self.eof = False
        self.fd = None 
        self.width = None
        self.height = None

        if not mode in EventReader.READING_MODES:
            raise ValueError(f"Mode {mode} not supported. Supported modes are: {EventReader.READING_MODES}")
        self.mode = mode

        # if mode is auto, we will try to infer the mode from the parameters
        if self.mode == "auto":
            if delta_t is not None and n_events is not None:
                self.mode = "mixed"
            elif delta_t is not None:
                self.mode = "delta_t"
                n_events = -1
            elif n_events is not None:
                self.mode = "n_events"
                delta_t = -1
            else:
                delta_t = -1
                n_events = -1
                self.mode = "mixed"
        elif self.mode == "delta_t":
            if delta_t is None:
                raise ValueError("delta_t must be specified")
        elif self.mode == "n_events":
            if n_events is None:
                raise ValueError("n_events must be specified")


        self.delta_t = delta_t if delta_t > 0 else 10000
        self.n_events = n_events if n_events > 0 else max_events
        self.max_events = max_events

        self.is_initialized = False

        self.n_read_events = 0

    
    def init(self):
        raise NotImplementedError
    
    def read(self, delta_t=None, n_events=None) -> tuple[np.ndarray, bool]:
        raise NotImplementedError
    
    def __enter__(self):
        return self
    
    def is_eof(self) -> bool:
        return self.eof

    def close(self):
        raise NotImplementedError

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def __repr__(self) -> str:
        if self.is_initialized:
            is_initialized_txt = "initialized"
        else:
            is_initialized_txt = "not initialized"
        return f"{self.__class__}(file={self.file} - {is_initialized_txt}, delta_t={self.delta_t}, n_events={self.n_events}, mode={self.mode})"
    
    def __len__(self) -> int:
        return self.n_read_events
    
    def __iter__(self):
        while not self.is_eof():
            yield self.read()

    
    def file_size(self) -> int:
        return os.stat(self.file).st_size

    def tell(self) -> int:
        if self.fd is None:
            return 0
        return self.fd.tell()
