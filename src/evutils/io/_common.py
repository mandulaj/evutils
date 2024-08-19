from abc import ABC, abstractmethod

import numpy as np
from pathlib import Path

import os

class EventFileReader_Base(ABC):
    def __init__(self, file: Path):
        self.file = file
        self.fd = None
        
        self.is_initialized = False



    @abstractmethod
    def read_chunk(self) -> np.ndarray:
        '''
        Read a chunk of events
        '''
        raise NotImplementedError
    
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
    
    def file_size(self) -> int:
        '''
        Get the size of the file in bytes

        Returns
        -------
        int
            The size of the file in bytes
        '''


        return os.stat(self.file).st_size
    
    def progress(self) -> int:
        '''
        Get the current progress in the file

        Returns
        -------
        int
            The current progress in the file 0-1
        '''
        return self.tell() / self.file_size() 

    
    def close(self):
        '''
        Close the file and release the resources
        '''
        if self.is_initialized and self.fd is not None:
            self.fd.close()


    def __exit__(self, exc_type, exc_value, traceback):
        self.close()


class EventFileWriter_Base(ABC):
    def __init__(self, file: Path):
        self.file = file
        self.fd = None

        self.n_written_events = 0
        self.is_initialized = False



    @abstractmethod
    def write(self, events: np.ndarray) -> np.ndarray:
        '''
        Read a chunk of events
        '''
        raise NotImplementedError
    
    @abstractmethod
    def flush(self):
        '''
        Flush the buffer to the file
        '''
        raise NotImplementedError
    

    def close(self):
        '''
        Close the file and release the resources
        '''
        if self.is_initialized and self.fd is not None:
            self.fd.close()


    def __exit__(self, exc_type, exc_value, traceback):
        self.close()


    def __len__(self) -> int:
        return self.n_written_events
    
    
    def __enter__(self):
        return self