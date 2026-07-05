
import io
import os
from pathlib import Path
from typing import Any, Tuple, Union

import numpy as np

from ..types import Event_dtype, EventArray

from . import decoders as ev_decoders
from ._source import ByteSource, make_source
from .buffer import EventAccumulator


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

        # Windowed reads decode straight into this reused accumulator (no
        # intermediate copy); only the returned window is copied out. Allocated
        # lazily on first read() so the read_all() fast path never pays for it.
        # Granularity of a single decode step, and a native fast path flag.
        self._buffer: EventAccumulator | None = None
        self._step = 1 << 20
        # Hold the largest window we might be asked for, plus room for one decode
        # step of overshoot before the front is rotated out. (np.empty is lazy, so
        # a large capacity only costs pages actually written.)
        self._acc_capacity = max(self._n_events, max_events) + 2 * self._step
        self._native_fill = hasattr(self._file_decoder, "parse_step")

        self._n_read_events = 0 # Number of events read (not includeing events stored in buffer)



    def init(self):
        '''
        Initialize the reader, can be used explicitly or implicitly by the read method.
        '''
        if self._is_initialized:
            return
        self._file_decoder.init()
        self._is_initialized = True

    def _pull(self, acc: EventAccumulator, delta_t: int, n_events: int) -> int:
        '''Pull more events into the accumulator, returning the number added
        (0 => end of stream). Native decoders decode straight into the
        accumulator's storage (no copy); others have their ``read_chunk`` output
        appended.'''
        dec = self._file_decoder
        if self._native_fill:
            while True:
                if dec.is_eof():
                    return 0
                ev, tr = acc.prepare(self._step)
                added = dec.parse_step(ev, tr)
                if added > 0:
                    return added
                if dec.is_eof():
                    return 0
                # else: consumed only state/timing words; step again.
        chunk = dec.read_chunk(delta_t, n_events)
        if len(chunk) == 0:
            return 0
        acc.append(chunk)
        return len(chunk)

    def read(self, delta_t:int|None=None, n_events:int|None=None) -> EventArray:
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
        EventArray
            An array with the events

        '''
        # If not initialized, initialize
        if not self._is_initialized:
            self.init()

        # Allocate the staging accumulator on first use.
        if self._buffer is None:
            self._buffer = EventAccumulator(self._acc_capacity)
        acc = self._buffer

        # Override the parameters if they are specified
        if delta_t is None:
            delta_t = self._delta_t
        if n_events is None:
            n_events = self._n_events

        # Establish the first timestamp once, at the very start of the stream.
        if self._n_read_events == 0 and len(acc) == 0:
            if self._pull(acc, delta_t, n_events) == 0:
                self._eof = True
                return EventArray.empty()
            self._first_ts = int(acc.t_window()[0])
            self._current_ts = self._first_ts

        start_ts: int = self._current_ts
        end_ts: int = start_ts + delta_t  # Final end_ts if we reach delta_t
        end_idx: int = len(acc)

        # Gather events until we hit the n_events count, the delta_t time window,
        # or the end of the stream. Work directly on the SoA `t` column.
        while True:
            # n_events cutoff
            if len(acc) > n_events:
                end_idx = n_events
                self._current_ts = int(acc.t_window()[n_events])
                break

            # time (delta_t) cutoff
            t = acc.t_window()
            if len(t) > 0 and t[-1] > end_ts:
                end_idx = int(np.searchsorted(t, end_ts))
                self._current_ts += delta_t
                break

            # Not enough buffered yet: pull more from the decoder.
            if self._pull(acc, delta_t, n_events) == 0:
                self._eof = True
                # Neither cutoff met, so the whole remaining buffer is the slice.
                end_idx = len(acc)
                break

        # Copy out the window (independent) and advance past it.
        output: EventArray = acc.slice_copy(end_idx)
        self._n_read_events += end_idx

        if self._normalize_ts:
            # Normalize the timestamps to start from zero at start_ts. slice_copy
            # already returned an independent array, so this is safe in place.
            output.t -= self._first_ts - self._start_ts
        return output

    def read_all(self) -> EventArray:
        '''
        Decode and return every remaining event at once.

        Delegates to the decoder's :meth:`~evutils.io.common.EventDecoder.read_all`,
        which (for the native EVT/DAT/AER decoders) decodes the whole payload
        straight into a single output buffer -- no per-chunk copy, no final
        ``concatenate`` -- and hands the columns back as a zero-copy
        :class:`EventArray`. This bypasses the slicing ring buffer that
        :meth:`read` uses for ``delta_t``/``n_events`` windowing.

        .. note::
            This materialises every event in memory at once. For recordings too
            large to fit, iterate the reader (windowed :meth:`read`) instead.

        Returns
        -------
        EventArray
            All remaining events.
        '''
        if not self._is_initialized:
            self.init()

        out = self._file_decoder.read_all()

        # Prepend anything already buffered by prior read() calls (rare: only if
        # read() and read_all() are mixed on the same reader).
        if self._buffer is not None and len(self._buffer) > 0:
            buffered = self._buffer.slice_copy(len(self._buffer))
            if len(out) == 0:
                out = buffered
            else:
                out = EventArray(
                    np.concatenate([buffered.t, out.t]),
                    np.concatenate([buffered.x, out.x]),
                    np.concatenate([buffered.y, out.y]),
                    np.concatenate([buffered.p, out.p]),
                )

        self._eof = True
        self._n_read_events += len(out)

        if self._normalize_ts and len(out) > 0:
            out.t -= int(out.t[0]) - self._start_ts

        return out

    def reset(self):
        '''Reset file reader back to the beginning of the file'''
        self._n_read_events = 0
        self._eof = False
        if self._buffer is not None:
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

        return self._eof and (self._buffer is None or len(self._buffer) == 0)

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
        EventArray
            An array with the events

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



