
import io
import os
from pathlib import Path
from typing import Any, Tuple, Union

import numpy as np

from ..types import Event_dtype

from . import decoders as ev_decoders
from ._source import ByteSource, make_source
from .buffer import EventRingBuffer


class EventReader():
    '''
    Class for reading and automatically decoding and slicing events from a file or stream.

    The reader supports different modes of operation, including reading a fixed number of events, reading events within a time window, or a combination of both. 
    The file_decoder is chosen automatically based on the file format or can be supplied explicitly.

    Parameters
    ----------
    file: Path or str or io.BufferedReader or bytes or ByteSource
        Path to the data file or a readable stream or bytes or a ByteSource
    delta_t: int or optional
        Time window in microseconds, by default None
    n_events: int or None
        Number of events to read in a chunk, by default None
    max_events: int, default=10_000_000
        Maximum number of events to read at once
    mode: {'delta_t', 'n_events', 'mixed', 'all', 'auto'}, default 'auto'
        Mode of operation ```["delta_t", "n_events", "mixed", "all", "auto"]```
    start_ts: int, default=0
        Start timestamp offset for the events, by default 0 (start of the file)
    normalize_ts: bool, default=False
        Normalize timestamps to start from zero
    max_time: int, default=1_000_000_000_000
        Maximum timestamp to read
    width: int or None
        Width of the frame, by default infered from the file
    height: int or None
        Height of the frame, by default infered from the file
    file_decoder: ev_decoders.EventDecoder or type[ev_decoders.EventDecoder] or None, default=None
        File decoder to use, by default None - automatic
    **kwargs
        Additional arguments to pass to the file decoder

    Raises
    ------
    ValueError
        If the mode is not supported or if the delta_t or n_events are not specified when needed

    Examples
    --------
    >>> for events in EventReader("events.raw", delta_t=10000):  
    >>>     print(events['x'], events['y'])

    '''

    READING_MODES = ["delta_t", "n_events", "mixed", "all", "auto"]
    DEFAULT_N_EVENTS = 1_000_000
    DEFAULT_DELTA_T = 10_000
    def __init__(self, file: Path | str | io.BufferedReader | bytes | ByteSource,
                 delta_t:int|None=None,
                 n_events:int|None=None,
                 mode:str="auto",
                 start_ts:int=0,
                 normalize_ts: bool=False,
                 max_time:int=1_000_000_000_000,
                 max_events:int=10_000_000,
                 width:int | None=None, height:int | None=None,
                 file_decoder: ev_decoders.EventDecoder | type[ev_decoders.EventDecoder] | None = None,
                 **kwargs):

        # Remember the path (if any) for repr / reset semantics.
        self._file_name: Path | None = Path(file) if isinstance(file, (str, Path)) else None

        # 1. Normalise the input into a ByteSource (path | stream | bytes |
        #    BytesIO | ByteSource -> ByteSource). Regular files are memory-mapped.
        self._source: ByteSource = make_source(file)

        # 2. Resolve the decoder and launch it:
        #    explicit instance > explicit class > heuristic (extension, then
        #    content sniffing of the source).
        if isinstance(file_decoder, ev_decoders.EventDecoder):
            self._file_decoder = file_decoder
        else:
            decoder_cls = file_decoder or ev_decoders.resolve_decoder_cls(self._source)
            self._file_decoder = decoder_cls(self._source, **kwargs)

        # This will now be io.BufferedReader
        self._eof = False

        # If not defined explicitly, the width and height are fetch from the file (not all formats support this)
        self._width = width
        self._height = height
        self._start_ts = start_ts # Offset to start reading events. 0 is start of file
        self._first_ts = 0 # First timestamp in the file, used for normalization
        self._current_ts = self._first_ts
        self._normalize_ts = normalize_ts # Normalize timestamps to start from zero



        # Validate the parameters
        if not mode in EventReader.READING_MODES:
            raise ValueError(f"Mode {mode} not supported. Supported modes are: {EventReader.READING_MODES}")


        self._mode = mode.lower()

        # if mode is auto, we will try to infer the mode from the parameters
        if self._mode == "auto":
            # If both delta_t and n_events are specified, we will use mixed mode
            if delta_t is not None and n_events is not None:
                self._mode = "mixed"

            # If only one of the parameters is specified, we will use that mode, the other will be set to the maximum
            elif delta_t is not None:
                self._mode = "delta_t"
                n_events = max_events
            elif n_events is not None:
                self._mode = "n_events"
                delta_t = max_time
            else:
                # If none of the parameters are specified, we will use the default Values
                self._mode = "mixed"
                delta_t = self.DEFAULT_DELTA_T
                n_events = self.DEFAULT_N_EVENTS

        # If the mode is not auto, we will check if the parameters are specified
        elif self._mode == "delta_t":
            if delta_t is None:
                raise ValueError("delta_t must be specified")
            n_events = max_events
        elif self._mode == "n_events":
            if n_events is None:
                raise ValueError("n_events must be specified")
            delta_t = max_time
        elif self._mode == "mixed":
            if delta_t is None:
                delta_t = self.DEFAULT_DELTA_T
            if n_events is None:
                n_events = self.DEFAULT_N_EVENTS

        elif self._mode == "all":
            delta_t = max_time
            n_events = max_events


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
        self._delta_t = delta_t
        self._n_events = n_events

        # Maximum number of events to read and maximum time to read in a chunk
        self._max_events = max_events if max_events < self._n_events else self._n_events
        self._max_time = max_time if max_time < self._delta_t else self._delta_t

        self._is_initialized = False

        # Internal buffer - must be initialized in the init method
        self._buffer = EventRingBuffer(2 * max_events)
        self._last_end_idx = 0

        self._n_read_events = 0 # Number of events read (not includeing events stored in buffer)



    def init(self):
        '''
        Initialize the reader, can be used explicitly or implicitly by the read method.
        '''
        if self._is_initialized:
            return
        self._file_decoder.init()
        self._is_initialized = True

    def read(self, delta_t:int|None=None, n_events:int|None=None) -> np.ndarray[Any, np.dtype[Any]]:
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
        if not self._is_initialized:
            self.init()

            

        # Override the parameters if they are specified
        if delta_t is None:
            delta_t = self._delta_t
        if n_events is None:
            n_events = self._n_events

        # print(f"Reading {n_events} events or {delta_t} microseconds")

        # This is done once at the beginning of the file read
        if self._n_read_events == 0 and len(self._buffer) == 0:
            # Fist skip the events that are before the start_ts            
            # TODO

            events_chunk = self._file_decoder.read_chunk(delta_t, n_events)
            if len(events_chunk) == 0:
                # We already reached the end of the file, must have been an empty file
                self._eof = True
                return np.array([], dtype=Event_dtype)
            

            self._first_ts = int(events_chunk[0]['t'])  # First timestamp in the file
            self._current_ts = self._first_ts

            self._buffer.append(events_chunk)
    
        
        start_ts: int = self._current_ts
        end_ts:int = start_ts + delta_t # Final end_ts if we raech delta_t
        end_idx:int = len(self._buffer) # Where do we slice the internal ring buffer?



        # 1. Try consuming the buffer first
        # 2. If the buffer is empty, read from the file

        # print("Buffer size:", len(self.buffer), "Start TS:", start_ts, "End TS:", end_ts, "N events:", n_events)

        # Gather events while we have less than delta_t time and n_events
        while True:

            # Check n_events condition
            if len(self._buffer) > n_events:
                end_idx = n_events
                self._current_ts = self._buffer.view()['t'][n_events]
                break

            # Check time condition
            if len(self._buffer) > 0 and self._buffer[-1]['t'] > end_ts:
                # print("We have enough time")
                t = self._buffer.view()['t'] # Does this need to be copies?
                end_idx = int(np.searchsorted(t, end_ts))

                # print(f"End idx by time: {end_idx}, {end_ts}, buffer size: {len(self.buffer)}, {t}")
                self._current_ts += delta_t
                break

            # We need more events, read from the file

            # TODO: Find better way of hinting the file_reader (delta_t, n_events)
            events_chunk = self._file_decoder.read_chunk(delta_t, n_events)
            # print(f"Read a Chunk of {len(events_chunk)} events, {self.n_read_events} total")
            # print(f"Buffer {len(self.buffer)} events")
            # print(f"Events chunk: {len(events_chunk)}")

            # Check if the file_reader has reached the end of the file
            if len(events_chunk) == 0:
                self._eof = True
                # Return everything gathered so far. We only get here when
                # neither the n_events nor the time cutoff was met, so the whole
                # (sub-n_events) buffer is the correct final slice -- including
                # events appended in earlier iterations of this loop.
                end_idx = len(self._buffer)
                break

            self._buffer.append(events_chunk)
            # print(f"Buffer size after append: {len(self.buffer)} end_idx: {end_idx}")


        # print(f"End idx {end_idx}, buffer size: {len(self.buffer)}")

        # print(f"End idx {end_idx}")
        # Grab the events to be returend and advance the buffers
        output_evbuffer = self._buffer[:end_idx].copy()
        self._buffer.advance(end_idx)
        self._n_read_events += end_idx


        if self._normalize_ts:
            # Normalize the timestamps to start from zero at start_ts
            output_evbuffer['t'] -= self._first_ts - self._start_ts
        return output_evbuffer

    def reset(self):
        '''Reset file reader back to the beginning of the file'''
        self._n_read_events = 0
        self._buffer.reset()
        self._file_decoder.reset()

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

        return self._eof and len(self._buffer) == 0

    def close(self):
        '''
        Close the reader and release resources (decoder buffer views, then the
        underlying byte source).
        '''
        # Drop decoder views (e.g. into an mmap) before closing the source.
        self._file_decoder.close()
        self._source.close()

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def __repr__(self) -> str:
        if self._is_initialized:
            is_initialized_txt = "initialized"
        else:
            is_initialized_txt = "not initialized"
        src = self._file_name if self._file_name is not None else self._source.__class__.__name__
        return f"{self.__class__.__name__}(source={src} - {is_initialized_txt}, delta_t={self._delta_t}, n_events={self._n_events}, mode={self._mode})"

    def __len__(self) -> int:
        return self._n_read_events

    def __iter__(self):
        '''
        Iterate over the events in the file

        Yields
        -------
        np.ndarray
            A numpy array with the events

        '''
        if not self._is_initialized:
            self.init()
        while not self.is_eof():
            yield self.read()

    def shape(self) -> tuple[int|None, int|None]:
        '''
        Get the shape of the frame

        Returns
        -------
        tuple[int, int]
            The shape of the frame (width, height)
        '''
        if self._width is not None and self._height is not None:
            return self._width, self._height
        else:
            return self._file_decoder.shape()


    def tell(self) -> int:
        '''
        Get the current position in the file

        Returns
        -------
        int
            The current position in the file
        '''
        return self._file_decoder.tell()



