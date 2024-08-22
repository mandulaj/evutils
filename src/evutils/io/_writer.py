
from datetime import datetime
import numpy as np

from typing import Union
from ._common import EventFileWriter_Base

from ..io import writer as ev_writers


from pathlib import Path

class EventWriter():
    '''
    Base class for writing events to different file formats

    Parameters
    ----------
    file
        Path to the data file
    width
        Width of the frame, by default 1280 (not relevant for some formats)
    height
        Height of the frame, by default 720 (not relevant for some formats)
    dt
        Timestamp of the recording (default is the current time, but information is not saved in all formats)
    file_writer
        File writer to use, by default 'auto'
    **kwargs
        Additional arguments for the file writer

    Examples
    --------
    >>> events = np.array([(0, 0, 0, 1), (1000, 10, 10, 0)], dtype=Event_dtype)
    >>> with EventWriter("events.raw", delta_t=10000) as writer:
    >>>     writer.write(events)
    '''
    def __init__(self, file:Union[Path, str], width:int=1280, height:int=720, dt: datetime = None,  file_writer: Union[EventFileWriter_Base, str]='auto', **kwargs):

        if not isinstance(file, Path):
            file = Path(file)
        self.file = file
        self.width = width
        self.height = height

        self.n_written_events = 0

        self.is_initialized = False

        if dt is None:
            self.dt = datetime.now()
        else:
            self.dt = dt

        # File writer for differnt file types
        if isinstance(file_writer, EventFileWriter_Base):
            self.file_writer = file_writer
        elif isinstance(file_writer, str) and file_writer == "auto":
            self.file_writer = self._create_file_writer(file, kwargs)
        else:
            raise ValueError("file_writer must be a EventFileWriter or 'auto'")

    
    def _create_file_writer(self, file_name:Path, args:dict={}) -> EventFileWriter_Base:
        '''
        Create the file writer based on the file extension

        Returns
        -------
        EventFileWriter_Base
            The file writer
        '''
        
        reader_cls = ev_writers.get_file_writer(file_name)

        return reader_cls(file_name, **args)

    def init(self):
        '''
        Initialize the writer (e.g. open the file, write the header)

        This method can be called explicitly, but it is also called automatically when the first event is written
        '''
        self.file_writer.init()
    
    def write(self, events: np.ndarray) -> int:
        '''
        Write a buffer of events to the file

        Parameters
        ----------
        events
            Buffer of events to write
        
        Returns
        -------
        int
            Number of events written
        '''
        return self.file_writer.write(events)
    
    def flush(self):
        '''
        Flush the buffer to the file
        '''
        self.file_writer.flush()

    def __enter__(self):
        return self
    
    def close(self):
        '''
        Close the file and release the resources
        '''
        self.file_writer.close()
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