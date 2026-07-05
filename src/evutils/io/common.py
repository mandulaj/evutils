import io
import os
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import IO, Any, Optional

import numpy as np




class EventDecoder(ABC):
    '''
    ABC for reading chunks of events from a IO source object

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
    def __init__(self, source, chunk_size:int = 10000):
        # `source` is a ByteSource (see io/_source.py). `fd` is kept as a legacy
        # alias for older decoders that still reference it.
        self._source = source
        self._fd = source

        self._is_initialized = False

        self._chunk_size = chunk_size

        self._eof = False

        self._width = None
        self._height = None

    @abstractmethod
    def init(self):
        '''
        Initialize the file for reading
        '''
        raise NotImplementedError

    @abstractmethod
    def read_chunk(self, delta_t_hint:int | None = None, n_events_hint:int | None = None) -> np.ndarray[Any, np.dtype[Any]]:
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
    def reset(self):
        '''
        Reset the file pointer to the beginning of the file
        '''
        raise NotImplementedError


    def read_all(self):
        '''
        Decode and return every remaining event at once.

        The default implementation drains :meth:`read_chunk` and concatenates the
        chunks. SoA-native decoders (EVT/DAT/AER) override this with a
        single-buffer decode that avoids the per-chunk copy entirely.

        Returns
        -------
        EventArray
            All remaining events.
        '''
        from ..types import EventArray

        if not self._is_initialized:
            self.init()

        # read_chunk may return a view that is invalidated by the next call, so
        # copy each chunk before pulling the next one.
        chunks = []
        while True:
            chunk = self.read_chunk()
            if len(chunk) == 0:
                break
            chunks.append(chunk.copy())

        if not chunks:
            return EventArray.empty()
        if len(chunks) == 1:
            return chunks[0]
        return EventArray(
            np.concatenate([c.t for c in chunks]),
            np.concatenate([c.x for c in chunks]),
            np.concatenate([c.y for c in chunks]),
            np.concatenate([c.p for c in chunks]),
        )

    def close(self):
        '''Release any resources held by the decoder (e.g. buffer views).

        The owning source is closed separately by the EventReader.
        '''
        pass

    def tell(self) -> int:
        '''
        Get the current position in the file

        Returns
        -------
        int
            The current position in the file
        '''
        return self._fd.tell()


    def set_chunk_size(self, chunk_size:int):
        '''
        Set the chunk size

        Parameters
        ----------
        chunk_size
            Size of the chunk to read
        '''
        self._chunk_size = chunk_size

    def shape(self) -> tuple[int|None, int|None]:
        '''
        Get the shape of the frame (width, height)

        Returns
        -------
        tuple[int|None, int|None]
                The shape of the frame (width, height), or (None, None) if the shape is not known
        '''
        return self._width, self._height
        


    def __repr__(self) -> str:
            if self._is_initialized:
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
        return self._eof




class EventEncoder(ABC):
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

        self._fd = writable

        self._width = width
        self._height = height


        self._n_written_events = 0
        self._is_initialized = False

        if dt is None:
            self._dt = datetime.now()
        else:
            self._dt = dt
    
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

    def __len__(self) -> int:
        return self._n_written_events


    def __enter__(self):
        return self


    def __repr__(self) -> str:
        if self._is_initialized:
            is_initialized_txt = f"Written {self._n_written_events} events"
        else:
            is_initialized_txt = "not initialized"
        return f"{self.__class__.__name__} - {is_initialized_txt}, {self._width}x{self._height})"

    def flush(self):
        self._fd.flush()
