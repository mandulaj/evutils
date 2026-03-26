import io
import os
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import IO, Any, Optional

import numpy as np


class EventDecoder_Base(ABC):
    @abstractmethod
    def init(self):
        '''
        Initialize the file for reading
        '''
        raise NotImplementedError

    @abstractmethod
    def read_chunk(self, delta_t_hint:int|None = None, n_events_hint:int|None = None) -> np.ndarray[Any, np.dtype[Any]]:
        '''
        Read a chunk of events

        Parameters
        ----------
        delta_t_hint
            If not None, can be used to provide a hit about the delta_t window to be read
        n_events_hint
            If not None, can be used to provide a hit about the n_events to be read
        '''
        raise NotImplementedError

    @abstractmethod
    def tell(self) -> int:
        '''
        Get the current position in the file

        Returns
        -------
        int
            The current position in the file
        '''
        raise NotImplementedError

    @abstractmethod
    def reset(self):
        '''
        Reset the file pointer to the beginning of the file
        '''
        raise NotImplementedError


class EventDecoder(EventDecoder_Base):
    '''
    ABC for reading chunks of events from a file format

    Parameters
    ----------
    readable
        source to read events from
    chunk_size
        Size of the chunk to read

    Raises
    ------

    NotImplementedError
        If the method is not implemented in the subclass

    '''
    def __init__(self, readable: io.BufferedReader, chunk_size:int):
        self.fd = readable

        self.is_initialized = False

        self.chunk_size = chunk_size

        self.eof = False

        self.width = None
        self.height = None

    def tell(self) -> int:
        '''
        Get the current position in the file

        Returns
        -------
        int
            The current position in the file
        '''
        return self.fd.tell()


    def set_chunk_size(self, chunk_size:int):
        '''
        Set the chunk size

        Parameters
        ----------
        chunk_size
            Size of the chunk to read
        '''
        self.chunk_size = chunk_size

    def shape(self) -> tuple[int|None, int|None]:
        '''
        Get the shape of the frame (width, height)

        Returns
        -------
        tuple[int|None, int|None]
                The shape of the frame (width, height), or (None, None) if the shape is not known
        '''
        return self.width, self.height
        


    def __repr__(self) -> str:
            if self.is_initialized:
                is_initialized_txt = "initialized"
            else:
                is_initialized_txt = "not initialized"
            return f"{self.__class__} - {is_initialized_txt})"

    def is_eof(self) -> bool:
        '''
        Check if the end of the file has been reached

        Returns
        -------
        bool
            True if the end of the file has been reached
        '''
        return self.eof

class EventEncoder_Base(ABC):
    @abstractmethod
    def init(self):
        '''
        Initialize the file for writing
        '''
        raise NotImplementedError

    @abstractmethod
    def write(self, events: np.ndarray[Any, np.dtype[Any]]) -> int:
        '''
        Write a chunk of events
        '''
        raise NotImplementedError

    @abstractmethod
    def flush(self):
        '''
        Flush the buffer to the file
        '''
        raise NotImplementedError



class EventEncoder(EventEncoder_Base):
    '''
    ABC for writing chunks of events to a io object

    Parameters
    ----------
    writable
        Destination for writing events
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
    def __init__(self, writable: io.BufferedWriter, width:int = 1280, height:int = 720, dt:Optional[datetime]=None ):

        self.fd = writable

        self.width = width
        self.height = height


        self.n_written_events = 0
        self.is_initialized = False

        if dt is None:
            self.dt = datetime.now()
        else:
            self.dt = dt


    def __len__(self) -> int:
        return self.n_written_events


    def __enter__(self):
        return self


    def __repr__(self) -> str:
        if self.is_initialized:
            is_initialized_txt = f"Written {self.n_written_events} events"
        else:
            is_initialized_txt = "not initialized"
        return f"{self.__class__.__name__} - {is_initialized_txt}, {self.width}x{self.height})"

    def flush(self):
        self.fd.flush()
