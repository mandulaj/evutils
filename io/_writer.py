
from datetime import datetime
import numpy as np

class EventWriter():
    def __init__(self, file, width=1280, height=720, dt: datetime = None):

        self.file = file
        self.fd = None 
        self.width = width
        self.height = height

        

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