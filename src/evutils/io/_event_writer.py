"""Event writer module.

Provides the `EventWriter` class for writing event data to various file formats.
"""

import io
from datetime import datetime
from pathlib import Path

import numpy as np

from . import encoders as ev_encoders

class EventWriter():
    """Base class for writing events to different file formats.

    Parameters
    ----------
    file
        Path to the data file
    width
        Width of the frame. If None (default), taken from the first written
        events' ``sensor_size`` metadata, falling back to 1280 (not relevant
        for some formats).
    height
        Height of the frame. If None (default), taken from the first written
        events' ``sensor_size`` metadata, falling back to 720 (not relevant
        for some formats).
    dt
        Timestamp of the recording (default is the current time, but information is not saved in all formats)
    file_encoder
        File encoder to use, by default None (chosen from the file extension)
    mode
        File open mode, by default 'w+b'
    **kwargs
        Additional arguments for the file encoder

    Examples
    --------
    >>> import numpy as np
    >>> from evutils.types import EventArray
    >>> # Create an EventArray
    >>> events = EventArray(
    ...     t=np.array([0, 1000]),
    ...     x=np.array([0, 10]),
    ...     y=np.array([0, 10]),
    ...     p=np.array([1, 0])
    ... )
    >>> # Write the events to a raw file
    >>> with EventWriter("events.raw") as writer: # doctest: +SKIP
    ...     writer.write(events) # doctest: +SKIP

    """

    def __init__(self, file: Path | str | io.BufferedIOBase, width:int|None=None, height:int|None=None, dt: datetime|None = None,  file_encoder: ev_encoders.EventEncoder | None = None, mode: str = 'w+b', **kwargs):
        self._mode = mode
        self._file_name: Path | None = None

        # Handle paths as input
        if isinstance(file, str):
            file = Path(file)
        if isinstance(file, Path):
            self._file_name = file
            file = self._open_file(file)
        else:
            # A raw stream was passed - we need an explicit encoder.
            if file_encoder is None:
                raise ValueError("When using a binary stream as file, the file_encoder must be provided explicitly")

        if isinstance(file, io.IOBase) and not file.writable():
            raise IOError("File is not writable")
        self._file: io.BufferedIOBase = file

        # Encoder resolution is deferred: when width/height are not given
        # explicitly we try to pick them up from the first written events'
        # ``sensor_size`` metadata (falling back to 1280x720). An explicitly
        # supplied encoder instance is used as-is.
        self._explicit_width = width
        self._explicit_height = height
        self._raw_dt = dt                       # passed verbatim to the encoder
        self._encoder_kwargs = kwargs
        self._file_encoder = file_encoder       # None => built lazily

        self._width = width
        self._height = height
        self._n_written_events = 0
        self._is_initialized = False
        self._warned_triggers = False  # warn once about unsupported triggers
        self._dt = dt if dt is not None else datetime.now()

    def _ensure_encoder(self, events: "EventArray | None" = None) -> None:
        """Build the file encoder if it hasn't been resolved yet.

        Dimensions are resolved in priority order: explicit ``width``/``height``
        from the constructor, then the events' ``sensor_size`` metadata, then a
        ``1280x720`` fallback. Formats that ignore geometry are unaffected.
        """
        if self._file_encoder is not None:
            return
        assert self._file_name is not None
        w, h = self._explicit_width, self._explicit_height
        if w is None or h is None:
            ss = getattr(events, "sensor_size", None)
            if ss is not None:
                if w is None:
                    w = int(ss[0])
                if h is None:
                    h = int(ss[1])
        if w is None:
            w = 1280
        if h is None:
            h = 720
        self._width, self._height = w, h
        encoder_cls = ev_encoders.get_file_writer(self._file_name)
        self._file_encoder = encoder_cls(self._file, width=w, height=h,
                                         dt=self._raw_dt, **self._encoder_kwargs)

    def _open_file(self, file_name: Path) -> io.BufferedIOBase:
        """Open the file for writing.

        Parameters
        ----------
        file_name : Path
            Path to the file to open.

        Returns
        -------
        io.BufferedIOBase
            The opened file object.

        """
        # default 'w+b' (not 'wb'): container encoders (HDF5) need the stream to be
        # readable and seekable, and it costs nothing for the append-only ones.
        return open(str(file_name), self._mode)

    def init(self) -> None:
        """Initialize the writer (e.g. open the file, write the header).

        This method can be called explicitly, but it is also called automatically when the first event is written
        """
        self._ensure_encoder(None)
        self._file_encoder.init()
        self._is_initialized = True

    def write(self, events: np.ndarray, triggers: np.ndarray | None = None) -> int:
        """Write a buffer of events (and optionally external triggers) to the file.

        Parameters
        ----------
        events
            Buffer of events to write (structured array or EventArray)
        triggers
            Buffer of triggers to write (structured array or TriggerArray). Optional.
            Interleaved with the events by timestamp for formats that carry
            triggers in-stream (EVT); ignored (with a warning from the encoder)
            by formats that cannot represent them. To write a trigger-only
            batch, pass ``EventArray.empty()`` as ``events``.

        Returns
        -------
        int
            Number of events written

        Examples
        --------
        >>> from evutils.types import EventArray
        >>> events = EventArray(t=[0, 100], x=[10, 11], y=[20, 21], p=[1, 0])
        >>> writer = EventWriter("events.raw") # doctest: +SKIP
        >>> num_written = writer.write(events) # doctest: +SKIP
        >>> print(f"Wrote {num_written} events") # doctest: +SKIP
        >>> writer.close() # doctest: +SKIP

        """
        # Resolve the encoder lazily so the first batch can supply sensor_size.
        if self._file_encoder is None:
            self._ensure_encoder(events)

        if (triggers is not None and len(triggers) > 0
                and not getattr(self._file_encoder, "SUPPORTS_WRITE_TRIGGERS", False)
                and not self._warned_triggers):
            import warnings
            warnings.warn(
                f"{self._file_encoder.__class__.__name__} does not support "
                f"writing external triggers; they will NOT be stored.",
                stacklevel=2,
            )
            self._warned_triggers = True
        n_written = self._file_encoder.write(events, triggers=triggers)
        self._n_written_events += n_written
        return n_written

    def flush(self) -> None:
        """Flush the buffer to the file."""
        if self._file_encoder is not None:
            self._file_encoder.flush()

    def __enter__(self) -> "EventWriter":
        return self

    def __repr__(self) -> str:
        if self._is_initialized:
            is_initialized_txt = f"Written {self._n_written_events} events"
        else:
            is_initialized_txt = "not initialized"
        return f"{self.__class__.__name__}(file={self._file} - {is_initialized_txt}, {self._width}x{self._height})"

    def __len__(self) -> int:
        return self._n_written_events

    def close(self) -> None:
        """Close the writer and release the resources.

        Finalizes the encoder first (container formats write their archive /
        index here), then closes the underlying file.
        """
        # A writer that was opened but never written to still gets a valid
        # (header-only) file, matching the pre-lazy-init behaviour.
        self._ensure_encoder(None)
        self._file_encoder.close()
        if self._file_name is not None:
            self._file.close()

    def __exit__(self, exc_type: "type[BaseException] | None", exc_value: "BaseException | None", traceback: "types.TracebackType | None") -> None:
        self.close()
