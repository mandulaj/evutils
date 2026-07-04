
import io
import os
from pathlib import Path
from typing import Any, Tuple, Union

import numpy as np

from ..io import _decoders as ev_decoders
from ..types import Event_dtype
from ._common import EventDecoder_Base
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
    normalize_ts
        Normalize timestamps to start from zero, by default False
    max_time
        Maximum timestamp to read
    width
        Width of the frame, by default infered from the file
    height
        Height of the frame, by default infered from the file
    file_decoder
        File decoder to use, by default None - automatic
    **kwargs
        Additional arguments to pass to the file decoder

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
    def __init__(self, file: Path | str | io.BufferedReader,
                 delta_t:int|None=None,
                 n_events:int|None=None,
                 mode:str="auto",
                 start_ts:int=0,
                 normalize_ts: bool=False,
                 max_time:int=1_000_000_000_000,
                 max_events:int=10_000_000,
                 width:int | None=None, height:int | None=None,
                 file_decoder: EventDecoder_Base | None = None,
                 **kwargs):


        self.file_name:Path|None = None

        # Handle paths as input
        # if file is not a Path, convert it to a Path
        if isinstance(file, str):
            file = Path(file)
        if isinstance(file, Path):
            if not file.exists() or not file.is_file():
                raise FileNotFoundError(f"File {file} does not exist")

            self.file_name = file


            file = self._open_file(file)

        else:
            # File was passed a io.BufferedReader - we need an explicit file_decoder
            if file_decoder is None:
                raise ValueError(f"When using a io.BufferedReader as file, the file_decoder must be provided explicitly")

        if isinstance(file, io.BufferedReader):
            if not file.readable():
                raise IOError("File is not readable")
        self.file: io.BufferedReader = file

        # File decoder for differnt file types
        if file_decoder is None:
            assert self.file_name is not None
            self.file_decoder = self._create_file_decoder(self.file_name, kwargs)
        else:
            self.file_decoder = file_decoder

        # This will now be io.BufferedReader
        self.eof = False

        # If not defined explicitly, the width and height are fetch from the file (not all formats support this)
        self.width = width
        self.height = height
        self.start_ts = start_ts # Offset to start reading events. 0 is start of file
        self.first_ts = 0 # First timestamp in the file, used for normalization
        self.current_ts = self.first_ts
        self.normalize_ts = normalize_ts # Normalize timestamps to start from zero



        # Validate the parameters
        if not mode in EventReader.READING_MODES:
            raise ValueError(f"Mode {mode} not supported. Supported modes are: {EventReader.READING_MODES}")


        self.mode = mode.lower()

        # if mode is auto, we will try to infer the mode from the parameters
        if self.mode == "auto":
            # If both delta_t and n_events are specified, we will use mixed mode
            if delta_t is not None and n_events is not None:
                self.mode = "mixed"

            # If only one of the parameters is specified, we will use that mode, the other will be set to the maximum
            elif delta_t is not None:
                self.mode = "delta_t"
                n_events = max_events
            elif n_events is not None:
                self.mode = "n_events"
                delta_t = max_time
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
        self.delta_t = delta_t
        self.n_events = n_events

        # Maximum number of events to read and maximum time to read in a chunk
        self.max_events = max_events if max_events < self.n_events else self.n_events
        self.max_time = max_time if max_time < self.delta_t else self.delta_t

        self.is_initialized = False

        # Internal buffer - must be initialized in the init method
        self.buffer = EventRingBuffer(2 * max_events)
        self.last_end_idx = 0

        self.n_read_events = 0 # Number of events read (not includeing events stored in buffer)



    def init(self):
        '''
        Initialize the reader, can be used explicitly or implicitly by the read method.
        '''
        if self.is_initialized:
            return
        self.file_decoder.init()
        self.is_initialized = True

    def _open_file(self, file_name: Path) -> io.BufferedReader:
        # TODO: Handle compressed files
        return open(str(file_name), 'rb')


    def _create_file_decoder(self, file_name: Path, args:dict={}) -> EventDecoder_Base:
        '''
        Create the file reader based on the file extension

        Returns
        -------
        EventFileReader_Base
            The file writer
        '''

        decoder_cls = ev_decoders.get_reader_from_filename(file_name)

        return decoder_cls(self.file, **args)




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
        if not self.is_initialized:
            self.init()

            

        # Override the parameters if they are specified
        if delta_t is None:
            delta_t = self.delta_t
        if n_events is None:
            n_events = self.n_events

        # print(f"Reading {n_events} events or {delta_t} microseconds")

        # This is done once at the beginning of the file read
        if self.n_read_events == 0 and len(self.buffer) == 0:
            # Fist skip the events that are before the start_ts            
            # TODO

            events_chunk = self.file_decoder.read_chunk(delta_t, n_events)
            if len(events_chunk) == 0:
                # We already reached the end of the file, must have been an empty file
                self.eof = True
                return np.array([], dtype=Event_dtype)
            

            self.first_ts = int(events_chunk[0]['t'])  # First timestamp in the file
            self.current_ts = self.first_ts

            self.buffer.append(events_chunk)
    
        
        start_ts: int = self.current_ts
        end_ts:int = start_ts + delta_t # Final end_ts if we raech delta_t
        end_idx:int = len(self.buffer) # Where do we slice the internal ring buffer?



        # 1. Try consuming the buffer first
        # 2. If the buffer is empty, read from the file

        # print("Buffer size:", len(self.buffer), "Start TS:", start_ts, "End TS:", end_ts, "N events:", n_events)

        # Gather events while we have less than delta_t time and n_events
        while True:

            # Check n_events condition
            if len(self.buffer) > n_events:
                end_idx = n_events
                self.current_ts = self.buffer.view()['t'][n_events]
                break

            # Check time condition
            if len(self.buffer) > 0 and self.buffer[-1]['t'] > end_ts:
                # print("We have enough time")
                t = self.buffer.view()['t'] # Does this need to be copies?
                end_idx = int(np.searchsorted(t, end_ts))

                # print(f"End idx by time: {end_idx}, {end_ts}, buffer size: {len(self.buffer)}, {t}")
                self.current_ts += delta_t
                break

            # We need more events, read from the file

            # TODO: Find better way of hinting the file_reader (delta_t, n_events)
            events_chunk = self.file_decoder.read_chunk(delta_t, n_events)
            # print(f"Read a Chunk of {len(events_chunk)} events, {self.n_read_events} total")
            # print(f"Buffer {len(self.buffer)} events")
            # print(f"Events chunk: {len(events_chunk)}")

            # Check if the file_reader has reached the end of the file
            if len(events_chunk) == 0:
                self.eof = True
                break

            self.buffer.append(events_chunk)
            # print(f"Buffer size after append: {len(self.buffer)} end_idx: {end_idx}")


        # print(f"End idx {end_idx}, buffer size: {len(self.buffer)}")

        # print(f"End idx {end_idx}")
        # Grab the events to be returend and advance the buffers
        output_evbuffer = self.buffer[:end_idx].copy()
        self.buffer.advance(end_idx)
        self.n_read_events += end_idx


        if self.normalize_ts:
            # Normalize the timestamps to start from zero at start_ts
            output_evbuffer['t'] -= self.first_ts - self.start_ts
        return output_evbuffer

    def reset(self):
        '''Reset file reader back to the beginning of the file'''
        self.n_read_events = 0
        self.buffer.reset()
        self.file_decoder.reset()

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
        if self.file_name is not None:
            self.file.close()

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

    def shape(self) -> tuple[int|None, int|None]:
        '''
        Get the shape of the frame

        Returns
        -------
        tuple[int, int]
            The shape of the frame (width, height)
        '''
        if self.width is not None and self.height is not None:
            return self.width, self.height
        else:
            return self.file_decoder.shape()


    def tell(self) -> int:
        '''
        Get the current position in the file

        Returns
        -------
        int
            The current position in the file
        '''
        return self.file_decoder.tell()



