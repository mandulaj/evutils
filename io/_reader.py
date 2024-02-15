
import numpy as np



class EventReader():
    READING_MODES = ["delta_t", "n_events", "mixed", "all"]
    def __init__(self, file, delta_t=10000, n_events=10000, mode="delta_t"):
        self.file = file
        self.eof = False
        self.fd = None 
        self.width = None
        self.height = None

        if not mode in EventReader.READING_MODES:
            raise ValueError(f"Mode {mode} not supported. Supported modes are: {EventReader.READING_MODES}")
        self.mode = mode
        self.delta_t = delta_t if delta_t > 0 else 10000
        self.n_events = n_events if n_events > 0 else 10000

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
        raise NotImplementedError