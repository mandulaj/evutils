
import numpy as np
from typing import Tuple, Union, Any

from pathlib import Path

from ..types import Event_dtype

import os


from ..io import reader as ev_readers


from ._common import EventFileReader_Base


from .buffer import EventRingBuffer



class EventReader():
    '''
    Class for reading events from different file formats

    Parameters
    ----------
    file 
        Path to the data file
    delta_t
        Time window in microseconds, by default None
    n_events
        Number of events to read in a chunk, by default None
    max_events
        Maximum number of events to read at once, by default 10,000,000
    mode
        Mode of operation ```["delta_t", "n_events", "mixed", "all", "auto"]```
    start_ts
        Start timestamp offset for the events, by default 0 (start of the file)
    max_time
        Maximum timestamp to read
    width
        Width of the frame, by default infered from the file
    height
        Height of the frame, by default infered from the file
    file_reader
        File reader to use, by default "auto"
    **kwargs
        Additional arguments to pass to the file reader

    Raises
    ------
    ValueError
        If the mode is not supported or if the delta_t or n_events are not specified when needed
    
    Examples
    --------
    >>> with EventReader("events.raw", delta_t=10000) as reader:
    >>>     for events in reader:
    >>>         print(events['x'], events['y'])
    
    '''
    READING_MODES = ["delta_t", "n_events", "mixed", "all", "auto"]
    DEFAULT_N_EVENTS = 1_000_000
    DEFAULT_DELTA_T = 10_000
    def __init__(self, file: Path | str,
                 delta_t:int=None, 
                 n_events:int=None,  
                 mode:str="auto", 
                 start_ts:int=0, 
                 max_time:int=1_000_000_000_000, 
                 max_events:int=10_000_000, 
                 width:int=None, height:int=None,
                 file_reader: Union[EventFileReader_Base, str]='auto',
                 **kwargs):

        # if file is not a Path, convert it to a Path
        if not isinstance(file, Path):
            file = Path(file)
        self.file = file
        self.eof = False
    
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
        
        if n_events > self.max_events:
            self.max_events = n_events + 10_000
        
        # delta_t and n_events to read on each call
        self.delta_t = delta_t
        self.n_events = n_events

        self.is_initialized = False

        # Internal buffer - must be initialized in the init method
        self.buffer = EventRingBuffer(max_events)
        self.last_end_idx = 0

        self.n_read_events = 0 # Number of events read (not includeing events stored in buffer)

        # File feader for differnt file types
        if isinstance(file_reader, EventFileReader_Base):
            self.file_reader = file_reader
        elif isinstance(file_reader, str) and file_reader == "auto":
            self.file_reader = self._create_file_reader(self.file, kwargs)
        else:
            raise ValueError("file_reader must be a EventFileReader or 'auto'")

    
    def init(self):
        '''
        Initialize the reader, can be used explicitly or implicitly by the read method.
        '''
        if self.is_initialized:
            return
        self.file_reader.init()
        self.is_initialized = True
    
    def _create_file_reader(self, file_name:Path, args:dict={}) -> EventFileReader_Base:
        '''
        Create the file reader based on the file extension

        Returns
        -------
        EventFileReader_Base
            The file writer
        '''
        
        reader_cls = ev_readers.get_file_reader(file_name)

        return reader_cls(file_name, **args)
    
        
    
   
    def read(self, delta_t:int=None, n_events:int=None) -> np.ndarray:
        '''
        Read events on the files based on the mode and the parameters
        
        Parameters
        ----------
        delta_t
            Override the delta_t parameter, otherwise the default value is used from the constructor
        n_events
            Override the n_events parameter, otherwise the default value is used from the constructor

        Returns
        ------- 
        np.ndarray
            A numpy array with the events of type :py:data:`Event_dtype`

        '''
        # If not initialized, initialize
        if not self.is_initialized:
            self.init()

        # Override the parameters if they are specified
        if delta_t is None:
            delta_t = self.delta_t
        if n_events is None:
            n_events = self.n_events

        # print(f"Reading {n_events} events or {delta_t} microseconds")

        start_ts = 0 if len(self.buffer) == 0 else self.buffer[0]['t'] # Start timestamp for the events
        end_ts = start_ts + delta_t # Final end_ts if we raech delta_t
        end_idx = len(self.buffer) # Where do we slice the internal ring buffer?
        

        # 1. Try consuming the buffer first
        # 2. If the buffer is empty, read from the file


        # Gather events while we have less than delta_t time and n_events
        while True:

            # Check n_events condition
            if len(self.buffer) > n_events:
                end_idx = n_events
                break

            if len(self.buffer) > 0 and self.buffer[-1]['t'] > end_ts:
                # print("We have enough time")
                t = self.buffer.view()['t'].copy()
                end_idx = np.searchsorted(t, end_ts)
                break


            # TODO: Find better way of hinting the file_reader (delta_t, n_events)
            events_chunk = self.file_reader.read_chunk(delta_t, n_events)
            # print(f"Read a Chunk of {len(events_chunk)} events, {self.n_read_events} total")
            # print(f"Buffer {len(self.buffer)} events")

            # Check if the file_reader has reached the end of the file
            if len(events_chunk) == 0:
                self.eof = True
                break

            end_idx += len(events_chunk) 
            self.buffer.append(events_chunk)


            
            
        
        # print(f"End idx {end_idx}")
        # Grab the events to be returend and advance the buffers
        output_evbuffer = self.buffer[:end_idx].copy()
        self.buffer.advance(end_idx)
        self.n_read_events += end_idx

        # print(f"Returning {len(output_evbuffer)} events, TOTAL: {self.n_read_events}")
        # print(f"{self.buffer.start} {self.buffer.end}")

        return output_evbuffer

    def reset(self):
        '''Reset file reader back to the beginning of the file'''
        self.n_read_events = 0
        self.buffer.reset()
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

        return self.eof and len(self.buffer) == 0

    def close(self):
        '''
        Close the file reader and release the resources
        '''
        print("Closing")
        self.file_reader.close()

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def __repr__(self) -> str:
        if self.is_initialized:
            is_initialized_txt = "initialized"
        else:
            is_initialized_txt = "not initialized"
        return f"{self.__class__.__name__}(file={self.file} - {is_initialized_txt}, delta_t={self.delta_t}, n_events={self.n_events}, mode={self.mode})"
    
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
            yield self.read()

    def shape(self) -> tuple[int, int]:
        '''
        Get the shape of the frame

        Returns
        -------
        tuple[int, int]
            The shape of the frame (width, height)
        '''
        return self.width, self.height
    
    def file_size(self) -> int:
        '''
        Get the size of the file in bytes

        Returns
        -------
        int
            The size of the file in bytes
        '''
        return self.file_reader.file_size()

    def tell(self) -> int:
        '''
        Get the current position in the file

        Returns
        -------
        int
            The current position in the file
        '''
        return self.file_reader.tell()
    
    def progress(self) -> int:
        '''
        Get the current progress in the file

        Returns
        -------
        int
            The current progress in the file 0-1
        '''
        return self.tell() / self.file_size() 
