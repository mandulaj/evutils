"""Common interface for event decoders and encoders.

Defines the abstract base classes `EventDecoder` and `EventEncoder`.
"""

import io
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from ..types import EventArray, TriggerArray

import numpy as np

class EventDecoder(ABC):
    """ABC for reading chunks of events from a IO source object.

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

    """

    # ------------------------------------------------------------------ #
    # Capability contract
    #
    # These class attributes declare which optional fast paths / features a
    # decoder supports. They all default to the conservative value here so a
    # decoder that does not override one degrades to the safe (correct, maybe
    # slower) path, and so the EventReader can read them directly instead of
    # probing with getattr(..., default) -- a missing attribute then surfaces
    # as a loud AttributeError rather than silently selecting the slow path.
    # Subclasses override the ones they implement.
    # ------------------------------------------------------------------ #

    #: Decoder can decode external trigger packets alongside CD events.
    SUPPORTS_EXT_TRIGGERS = False

    #: Whether this decoder implements timestamp / event-index random access via
    #: :meth:`seek`. Off by default; each seekable format opts in. The
    #: EventReader additionally requires the underlying ByteSource to be
    #: seekable before delegating a real (non-linear) seek here.
    SUPPORTS_SEEK = False

    #: Parser emits exactly one event per input record, so an output buffer
    #: fills to precisely its capacity -- enables EventReader's zero-copy
    #: n_events fast path. False for vectorised formats (EVT3/2.1/4).
    _exact_window = False

    #: read_chunk already returns slices of at most ``n_events`` events, so the
    #: reader can hand them straight through without re-accumulating (NPZ/HDF5).
    _independent_windows = False

    #: A dedicated C delta_t parser exists (one GIL-free call decodes a whole
    #: time window). EVT3 overrides this as a property.
    _has_delta_t_parser = False

    #: Seek-index wiring, injected by EventReader from its ``index=`` option.
    #: Only EVT consults them; harmless defaults for every other decoder.
    _use_sidecar = False
    _raw_path: "str | None" = None

    def __init__(self, source: "io.BufferedIOBase | str | bytes", chunk_size: int = 10000, read_external_triggers: bool = False):
        """Initialize the decoder.

        Parameters
        ----------
        source : ByteSource
            Source to read events from.
        chunk_size : int, optional
            Size of the chunk to read, by default 10000.
        read_external_triggers : bool, optional
            Whether to read external triggers, by default False.

        """
        # `source` is a ByteSource (see io/_source.py). `fd` is kept as a legacy
        # alias for older decoders that still reference it.
        self._source = source
        self._fd = source

        self._is_initialized = False

        self._chunk_size = chunk_size

        self._eof = False

        # Corrupt-packet policy (see EventReader(strict=...)): when True, a
        # malformed packet raises instead of being skipped with a warning.
        self._strict: bool = False

        self._width: int | None = None
        self._height: int | None = None

        self.read_external_triggers = read_external_triggers
        if self.read_external_triggers and not self.SUPPORTS_EXT_TRIGGERS:
            import warnings
            warnings.warn(f"{self.__class__.__name__} does not support reading external triggers.")

    @abstractmethod
    def init(self) -> None:
        """Initialize the file for reading."""
        raise NotImplementedError

    @abstractmethod
    def read_chunk(self, delta_t_hint:int | None = None, n_events_hint:int | None = None) -> 'EventArray | tuple[EventArray, TriggerArray]':
        """Read a chunk of events.

        Parameters
        ----------
        delta_t_hint : int, optional
            If not None, can be used to provide a hit about the delta_t window to be read
        n_events_hint : int, optional
            If not None, can be used to provide a hit about the n_events to be read
            
        Returns
        -------
        EventArray or tuple of (EventArray, TriggerArray)
            Chunk of events read from the source. Optionally returns triggers if `read_external_triggers` is True.

        """
        raise NotImplementedError

    @abstractmethod
    def reset(self) -> None:
        """Reset the file pointer to the beginning of the file."""
        raise NotImplementedError

    def take_pending(self) -> "EventArray | None":
        """Pop any boundary-chunk remainder staged by :meth:`seek`.

        Only decoders whose :meth:`seek` lands mid-chunk (EVT) stage a
        remainder; the base returns ``None`` so the EventReader can call this
        unconditionally.
        """
        return None

    def read_all(self) -> 'EventArray | tuple[EventArray, TriggerArray]':
        """Decode and return every remaining event at once.

        The default implementation drains :meth:`read_chunk` and concatenates the
        chunks. SoA-native decoders (EVT/DAT/AER) override this with a
        single-buffer decode that avoids the per-chunk copy entirely.

        Returns
        -------
        EventArray
            All remaining events.

        """
        from ..types import EventArray

        if not self._is_initialized:
            self.init()

        # read_chunk may return a view that is invalidated by the next call, so
        # copy each chunk before pulling the next one.
        chunks = []
        trigger_chunks = []
        while True:
            _chunk = self.read_chunk()
            if self.read_external_triggers:
                if isinstance(_chunk, tuple):
                    ev_chunk, tr_chunk = _chunk
                else:
                    ev_chunk = _chunk
                    from ..types import TriggerArray
                    tr_chunk = TriggerArray.empty()
            else:
                ev_chunk = _chunk # type: ignore
            if len(ev_chunk) == 0 and (not self.read_external_triggers or len(tr_chunk) == 0):
                break
            if len(ev_chunk) > 0:
                chunks.append(ev_chunk.copy())
            if self.read_external_triggers and len(tr_chunk) > 0:
                trigger_chunks.append(tr_chunk.copy())

        if not chunks:
            res_ev = EventArray.empty()
        elif len(chunks) == 1:
            res_ev = chunks[0]
        else:
            res_ev = EventArray(
                np.concatenate([c.t for c in chunks]),
                np.concatenate([c.x for c in chunks]),
                np.concatenate([c.y for c in chunks]),
                np.concatenate([c.p for c in chunks]),
            )
            
        if self.read_external_triggers:
            if not trigger_chunks:
                from ..types import TriggerArray
                res_tr = TriggerArray.empty()
            elif len(trigger_chunks) == 1:
                res_tr = trigger_chunks[0]
            else:
                from ..types import TriggerArray
                res_tr = TriggerArray(
                    np.concatenate([c.t for c in trigger_chunks]),
                    np.concatenate([c.p for c in trigger_chunks]),
                    np.concatenate([c.id for c in trigger_chunks]),
                )
            return res_ev, res_tr
            
        return res_ev

    def close(self) -> None:
        """Release any resources held by the decoder (e.g. buffer views).

        The owning source is closed separately by the EventReader.
        """
        pass

    def tell(self) -> int:
        """Get the current position in the file.

        Returns
        -------
        int
            The current position in the file

        """
        return int(self._fd.tell())

    @staticmethod
    def _seek_axis(t: int | None, n: int | None) -> "tuple[str, int]":
        """Validate the (t, n) pair and return the chosen axis and value.

        Returns ``("t", value)`` or ``("n", value)``; raises ``ValueError`` if
        neither or both were given.
        """
        if (t is None) == (n is None):
            raise ValueError("seek() requires exactly one of t= or n=.")
        if t is not None:
            return "t", int(t)
        return "n", int(n)  # type: ignore[arg-type]

    def seek(self, t: int | None = None, n: int | None = None) -> int:
        """Reposition the decoder to an absolute timestamp or event index.

        Exactly one of ``t`` (microseconds) or ``n`` (0-based event index) must
        be given. After a successful seek the next :meth:`read_chunk` /
        :meth:`read_all` yields events starting at the first event whose
        timestamp is ``>= t`` (time seek) or at event ``n`` (index seek).

        Parameters
        ----------
        t : int, optional
            Absolute target timestamp in microseconds.
        n : int, optional
            Absolute target event index (0-based).

        Returns
        -------
        int
            The absolute timestamp of the first event that will be read next
            (the landed position). Equals the stream's last timestamp + 1 style
            EOF sentinel handling is left to the caller; an empty tail simply
            means the target is at/after the end.

        Raises
        ------
        NotImplementedError
            If this decoder does not support seeking (``SUPPORTS_SEEK`` is
            False).
        ValueError
            If neither or both of ``t``/``n`` are provided.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support seek()."
        )

    def set_chunk_size(self, chunk_size:int) -> None:
        """Set the chunk size.

        Parameters
        ----------
        chunk_size
            Size of the chunk to read

        """
        self._chunk_size = chunk_size

    def shape(self) -> tuple[int|None, int|None]:
        """Get the shape of the frame (width, height).

        Returns
        -------
        tuple[int|None, int|None]
                The shape of the frame (width, height), or (None, None) if the shape is not known

        """
        return self._width, self._height
        

    def __repr__(self) -> str:
            if self._is_initialized:
                is_initialized_txt = "initialized"
            else:
                is_initialized_txt = "not initialized"
            return f"{self.__class__} - {is_initialized_txt}"

    def is_eof(self) -> bool:
        """Check if the end of the file has been reached.

        Returns
        -------
        bool
            True if the end of the file has been reached

        """
        return self._eof

class EventEncoder(ABC):
    """ABC for writing chunks of events to a io object.

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

    """

    #: Whether this encoder can write external triggers. No encoder implements
    #: trigger encoding yet; EventWriter warns (once) when triggers are passed
    #: to an encoder without support, instead of dropping them silently.
    SUPPORTS_WRITE_TRIGGERS = False

    def __init__(self, writable: io.BufferedIOBase, width:int = 1280, height:int = 720, dt:Optional[datetime]=None ):
        """Initialize the encoder.

        Parameters
        ----------
        writable : io.BufferedIOBase
            Destination for writing events.
        width : int, optional
            Width of the frame, by default 1280.
        height : int, optional
            Height of the frame, by default 720.
        dt : datetime, optional
            Timestamp of the recording, by default current time.

        """
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
    def init(self) -> None:
        """Initialize the file for writing."""
        raise NotImplementedError

    @abstractmethod
    def write(self, events: 'np.ndarray | EventArray', triggers: 'np.ndarray | TriggerArray | None' = None) -> int:
        """Write a chunk of events."""
        raise NotImplementedError

    def __len__(self) -> int:
        return self._n_written_events

    def __enter__(self) -> "EventEncoder":
        return self

    def __repr__(self) -> str:
        if self._is_initialized:
            is_initialized_txt = f"Written {self._n_written_events} events"
        else:
            is_initialized_txt = "not initialized"
        return f"{self.__class__.__name__} - {is_initialized_txt}, {self._width}x{self._height}"

    def flush(self) -> None:
        """Flush any buffered data to the underlying stream."""
        self._fd.flush()

    def close(self) -> None:
        """Finalize the encoder.

        Container formats (NPZ, HDF5) override this to write the archive /
        index before the underlying file is closed. The owning stream itself is
        closed by the :class:`~evutils.io.EventWriter`, not here.
        """
        self.flush()
