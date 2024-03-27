
import numpy as np

from pathlib import Path

import os

class EventReader():
    '''
    Base class for reading events from different file formats

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
    DEFAULT_N_EVENTS = 10_000
    DEFAULT_DELTA_T = 10_000
    def __init__(self, file, delta_t=None, n_events=None,  mode="auto", start_ts=0, max_time=1_000_000_000_000, max_events=10_000_000, width=None, height=None):
        print("EventReader constructor")

        self.file = Path(file)
        self.eof = False
        self.fd = None 
        self.width = width
        self.height = height
        self.start_ts = start_ts

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
        


        self.mode = mode

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
        

        self.delta_t = delta_t
        self.n_events = n_events

        self.is_initialized = False

        self.n_read_events = 0
        self.buffer_len = 0
        self.buffer = None

    
    def init(self):
        '''
        Initialize the reader, can be used explicitly or implicitly by the read method.
        '''
        raise NotImplementedError
    

    def _read(self, delta_t:int, n_events:int) -> np.ndarray:
        '''Reads the next n_events or delta_t events from the file, which ever comes first and returns them as a numpy array'''
        raise NotImplementedError
    
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

        # if self.mode == "delta_t":
        #     n_events = self.max_events
        #     self._read(delta_t, n_events)
        # elif self.mode == "n_events":
        #     delta_t = self.max_time
        # elif self.mode == "mixed":
        #     pass
        # elif self.mode == "all":
        #     pass

        return self._read(delta_t, n_events)


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
        '''
        Close the file and release the resources
        '''
        if self.is_initialized and self.fd is not None:
            self.fd.close()

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
            yield self._read(self.delta_t, self.n_events)

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
