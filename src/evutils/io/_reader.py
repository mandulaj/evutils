
import numpy as np

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
        '''
        Initialize the reader, can be used explicitly or implicitly by the read method.
        '''
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
        raise NotImplementedError
    
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
        '''
        Iterate over the events in the file
        
        Yields
        -------
        np.ndarray
            A numpy array with the events
        
        '''
        
        while not self.is_eof():
            yield self.read()

    
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
