
import numpy as np
from typing import Tuple, Union

from pathlib import Path

from ..types import Event_dtype

import os





class EventRingBuffer():
    def __init__(self, size:int, dtype=Event_dtype):
        self.size = size
        self.buffer = np.empty(size, dtype=dtype)
        self.dtype = dtype
        self.start = 0
        self.end = 0
    
    def __len__(self) -> int:
        return self.end - self.start
    
    @property
    def capacity(self) -> int:
        return self.size - self.end
    
    def append(self, data):
        if len(data) > self.capacity:
            self.rotate() # Try rotating to get more space

        if self.capacity > len(data):
            self.buffer[self.end:self.end+len(data)] = np.array(data, dtype=self.dtype)
            self.end += len(data)
        else:
            raise ValueError(f"Ring Buffer is full, can't append {len(data)} elememnts when {self.capacity} space left.")

    def advance(self, items:int):
        if self.start + items > self.size:
            raise ValueError(f"Cant advance beyond the buffer size {self.end}+{items}<{self.size}")
        self.start += items

    def view(self) -> np.ndarray:
        return self.buffer[self.start:self.end]
    
    def reset(self):
        self.start = 0
        self.end = 0

    def rotate(self):
        cur_len = len(self)
        self.buffer[0:cur_len] = self.view()
        self.start = 0
        self.end = cur_len

    def __getitem__(self, key):
        if isinstance(key, slice):
            
            # Deal if negative indexes
            start_idx = self.start + key.start if key.start >= 0 else self.end + key.start
            end_idx = self.start + key.stop if key.stop >= 0 else self.end + key.stop

            assert start_idx < end_idx

            return self.buffer[start_idx: end_idx: key.step]
        elif isinstance(key, int):

            idx = self.start + key if key >= 0 else self.end + key
            
            return self.buffer[idx]
        else:
            raise ValueError(f"Unsupported key in Ringbuffer {type(key)}")

        

class EventReader():
    '''
    Class for reading events from different file formats

    Parameters
    ----------
    file : str
        Path to the data file
    delta_t : int, optional
        Time window in microseconds, by default None
    n_events : int, optional
        Number of events to read in a chunk, by default None
    max_events : int, optional
        Maximum number of events to read at once, by default 10,000,000
    mode : {"auto", "delta_t", "n_events", "mixed", "all" }, optional
        Mode of operation, by default "auto"
    
    
    Raises
    ------
    ValueError
        If the mode is not supported or if the delta_t or n_events are not specified when needed
    NotImplementedError
        If the method is not implemented in the subclass

    '''
    READING_MODES = ["delta_t", "n_events", "mixed", "all", "auto"]
    DEFAULT_N_EVENTS = 1_000_000
    DEFAULT_DELTA_T = 10_000
    def __init__(self, file: Union[str, Path], 
                 delta_t:int=None, 
                 n_events:int=None,  
                 mode:str="auto", 
                 start_ts:int=0, 
                 max_time:int=1_000_000_000_000, 
                 max_events:int=10_000_000, 
                 width:int=None, height:int=None):

        # if file is not a Path, convert it to a Path
        if not isinstance(file, Path):
            file = Path(file)
        self.file = file
    
        self.width = width
        self.height = height
        self.start_ts = start_ts # Offset to start reading events. 0 is start of file

        # Maximum number of events to read and maximum time to read in a chunk
        self.max_events = max_events
        self.max_time = max_time


        # Validate the parameters
        if not mode in EventReader.READING_MODES:
            raise ValueError(f"Mode {mode} not supported. Supported modes are: {EventReader.READING_MODES}")
        
        if not self.file.exists() or not self.file.is_file():
            raise FileNotFoundError(f"File {self.file} does not exist")
        
        if not isinstance(self.max_events, int):
            raise TypeError("max_events must be an integer")
        
        if not isinstance(self.max_time, int):
            raise TypeError("max_time must be an integer")
        
        if not isinstance(self.start_ts, int):
            raise TypeError("start_ts must be an integer")
        

        self.mode = mode.lower()

        # if mode is auto, we will try to infer the mode from the parameters
        if self.mode == "auto":
            # If both delta_t and n_events are specified, we will use mixed mode
            if delta_t is not None and n_events is not None:
                self.mode = "mixed"

            # If only one of the parameters is specified, we will use that mode, the other will be set to the maximum
            elif delta_t is not None:
                self.mode = "delta_t"
                n_events = self.max_events
            elif n_events is not None:
                self.mode = "n_events"
                delta_t = self.max_time
            else:
                # If none of the parameters are specified, we will use the default Values
                self.mode = "mixed"
                delta_t = self.DEFAULT_DELTA_T
                n_events = self.DEFAULT_N_EVENTS

        # If the mode is not auto, we will check if the parameters are specified
        elif self.mode == "delta_t":
            if delta_t is None:
                raise ValueError("delta_t must be specified")
            n_events = self.max_events
        elif self.mode == "n_events":
            if n_events is None:
                raise ValueError("n_events must be specified")
            delta_t = self.max_time
        elif self.mode == "mixed":
            if delta_t is None:
                delta_t = self.DEFAULT_DELTA_T  
            if n_events is None:
                n_events = self.DEFAULT_N_EVENTS

        elif self.mode == "all":
            delta_t = self.max_time
            n_events = self.max_events

        
        # Validate the parameters
        if delta_t is None:
            delta_t = self.DEFAULT_DELTA_T
        if n_events is None:
            n_events = self.DEFAULT_N_EVENTS

        if not isinstance(delta_t, int):
            raise TypeError("delta_t must be an integer")

        if not isinstance(n_events, int):
            raise TypeError("n_events must be an integer")

        
        if delta_t <= 0:
            raise ValueError("delta_t must be positive")
        
        if n_events <= 0:
            raise ValueError("n_events must be positive")
        
        # delta_t and n_events to read on each call
        self.delta_t = delta_t
        self.n_events = n_events

        self.is_initialized = False

        # Internal buffer - must be initialized in the init method
        self.buffer = EventRingBuffer(max_events)

        self.n_read_events = 0 # Number of events read (not includeing events stored in buffer)

        # File feader for differnt file types
        self.file_reader = self._create_file_reader()

    
    def init(self):
        '''
        Initialize the reader, can be used explicitly or implicitly by the read method.
        '''
        self.file_reader.init()
    
    def _create_file_reader(self):
        reader_mapping = {
            # ".raw": ,
        }

        return reader_mapping
    
   
    def read(self, delta_t:int=None, n_events:int=None) -> np.ndarray:
        '''
        Read a chunk of events from the file
        
        Parameters
        ----------
        delta_t : int, optional
            Override the delta_t parameter, otherwise the default value is used from the constructor
        n_events : int, optional
            Override the n_events parameter, otherwise the default value is used from the constructor

        Returns
        ------- 
        np.ndarray
            A numpy array with the events

        Raises
        ------
        EOFError
            If the end of the file is reached

        '''
        # If not initialized, initialize
        if not self.is_initialized:
            self.init()

        # Override the parameters if they are specified
        if delta_t is None:
            delta_t = self.delta_t
        if n_events is None:
            n_events = self.n_events

        buf_to_send = None


        start_ts = 0
        end_ts = start_ts + delta_t # Final end_ts if we raech delta_t
        end_idx = 0 # Where do we slice the internal ring buffer?
        

        # Gather events while we have less than delta_t time and n_events
        while True:

            events_read = self.file_reader.read_chunk()
            if events_read == 0:
                # We have ran out of events to read
                break
            end_idx += events_read 

            # Check n_events condition
            if len(self.buffer) > n_events:
                end_idx = n_events
                break
            
            # # Check delta_t condition
            if self.buffer[-1]['t'] > end_ts:
                end_idx = np.searchsorted(self.buffer.view()['t'], end_ts)
                break

        # Grab the events to be returend and advance the buffers
        buf_to_send = self.buffer[:end_idx].copy()
        self.buffer.advance(end_idx)
        self.n_read_events += end_idx

        return buf_to_send

    def reset(self):
        '''Reset reader back to the beginning of the file'''
        self.file_reader.reset()

    def __enter__(self):
        return self
    
    def is_eof(self) -> bool:
        '''
        Check if the end of the file is reached
        
        Returns
        -------
        bool
            True if the end of the file is reached, False otherwise

        '''

        return self.eof

    def close(self):
        self.file_reader.close()

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
        '''
        Iterate over the events in the file
        
        Yields
        -------
        np.ndarray
            A numpy array with the events
        
        '''
        if not self.is_initialized:
            self.init()
        while not self.is_eof():
            try:
                yield self._read(self.delta_t, self.n_events)
            except StopIteration:
                break

    def shape(self) -> tuple[int, int]:
        return self.width, self.height
    
    def file_size(self) -> int:
        '''
        Get the size of the file in bytes

        Returns
        -------
        int
            The size of the file in bytes
        '''


        return os.stat(self.file).st_size

    def tell(self) -> int:
        '''
        Get the current position in the file

        Returns
        -------
        int
            The current position in the file
        '''
        if self.fd is None:
            return 0
        return self.fd.tell()
    
    def progress(self) -> int:
        '''
        Get the current progress in the file

        Returns
        -------
        int
            The current progress in the file 0-1
        '''
        return self.tell() / self.file_size() 
