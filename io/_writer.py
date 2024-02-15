
from datetime import datetime
import numpy as np

class EventWriter():
    def __init__(self, file, width=1280, height=720, dt: datetime = None):

        self.file = file
        self.fd = None 
        self.width = width
        self.height = height

        self.n_written_events = 0

        self.is_initialized = False

        if dt is None:
            self.dt = datetime.now()
        else:
            self.dt = dt


    def init(self):
        raise NotImplementedError
    
    def write(self, event: np.ndarray):
        raise NotImplementedError
    
    def __enter__(self):
        return self
    
    def close(self):
        raise NotImplementedError

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def __repr__(self) -> str:
        if self.is_initialized:
            is_initialized_txt = f"Written {self.n_written_events} events"
        else:
            is_initialized_txt = "not initialized"
        return f"{self.__class__.__name__}(file={self.file} - {is_initialized_txt}, {self.width}x{self.height})"
    
    def __len__(self) -> int:
        return self.n_written_events