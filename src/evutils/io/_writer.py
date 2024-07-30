
from datetime import datetime
import numpy as np

class EventWriter():
    '''
    Base class for writing events to different file formats

    Parameters
    ----------
    file : str
        Path to the data file
    width : int, optional
        Width of the frame, by default 1280 (not relevant for some formats)
    height : int, optional
        Height of the frame, by default 720 (not relevant for some formats)
    dt : datetime, optional
        Timestamp of the recording (default is the current time, but information is not saved in all formats)
    
    
    Raises
    ------
    NotImplementedError
        If the method is not implemented in the subclass

    '''
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
        '''
        Initialize the writer (e.g. open the file, write the header)

        This method can be called explicitly, but it is also called automatically when the first event is written
        '''
        raise NotImplementedError
    
    def write(self, events: np.ndarray):
        '''
        Write a buffer of events to the file

        Parameters
        ----------
        events : np.ndarray
            Buffer of events to write
        '''
        raise NotImplementedError
    
    def __enter__(self):
        return self
    
    def close(self):
        '''
        Close the file and release the resources
        '''
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