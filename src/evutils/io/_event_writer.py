"""Event writer module.

Provides the `EventWriter` class for writing event data to various file formats.
"""

import io
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from . import encoders as ev_encoders


class EventWriter():
    """Base class for writing events to different file formats.

    Parameters
    ----------
    file
        Path to the data file
    width
        Width of the frame, by default 1280 (not relevant for some formats)
    height
        Height of the frame, by default 720 (not relevant for some formats)
    dt
        Timestamp of the recording (default is the current time, but information is not saved in all formats)
    file_encoder
        File encoder to use, by default None (chosen from the file extension)
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

    def __init__(self, file: Path | str | io.BufferedIOBase, width:int=1280, height:int=720, dt: datetime|None = None,  file_encoder: ev_encoders.EventEncoder | None = None, **kwargs: Any):

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

        # Resolve the encoder: explicit instance > heuristic from extension.
        if file_encoder is None:
            assert self._file_name is not None
            encoder_cls = ev_encoders.get_file_writer(self._file_name)
            self._file_encoder = encoder_cls(self._file, width=width, height=height, dt=dt, **kwargs)
        else:
            self._file_encoder = file_encoder

        self._width = width
        self._height = height
        self._n_written_events = 0
        self._is_initialized = False
        self._warned_triggers = False  # warn once about unsupported triggers
        self._dt = dt if dt is not None else datetime.now()

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
        # 'w+b' (not 'wb'): container encoders (HDF5) need the stream to be
        # readable and seekable, and it costs nothing for the append-only ones.
        return open(str(file_name), 'w+b')

    def init(self) -> None:
        """Initialize the writer (e.g. open the file, write the header).

        This method can be called explicitly, but it is also called automatically when the first event is written
        """
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
        self._file_encoder.close()
        self._file.close()

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        self.close()
