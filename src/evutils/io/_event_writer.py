
import io
from datetime import datetime
from pathlib import Path
from typing import Union

import numpy as np

from . import encoders as ev_encoders


class EventWriter():
    '''
    Base class for writing events to different file formats

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
    >>> events = np.array([(0, 0, 0, 1), (1000, 10, 10, 0)], dtype=Event_dtype)
    >>> with EventWriter("events.raw") as writer:
    >>>     writer.write(events)
    '''
    def __init__(self, file: Path | str | io.BufferedWriter, width:int=1280, height:int=720, dt: datetime|None = None,  file_encoder: ev_encoders.EventEncoder | None = None, **kwargs):

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
                raise ValueError("When using a io.BufferedWriter as file, the file_encoder must be provided explicitly")

        if isinstance(file, io.BufferedWriter):
            if not file.writable():
                raise IOError("File is not writable")
        self._file: io.BufferedWriter = file

        # Resolve the encoder: explicit instance > heuristic from extension.
        if file_encoder is None:
            assert self._file_name is not None
            encoder_cls = ev_encoders.get_file_writer(self._file_name)
            self._file_encoder = encoder_cls(self._file, **kwargs)
        else:
            self._file_encoder = file_encoder

        self._width = width
        self._height = height
        self._n_written_events = 0
        self._is_initialized = False
        self._dt = dt if dt is not None else datetime.now()

    def _open_file(self, file_name: Path) -> io.BufferedWriter:
        return open(str(file_name), 'wb')

    def init(self):
        '''
        Initialize the writer (e.g. open the file, write the header)

        This method can be called explicitly, but it is also called automatically when the first event is written
        '''
        self._file_encoder.init()
        self._is_initialized = True

    def write(self, events: np.ndarray) -> int:
        '''
        Write a buffer of events to the file

        Parameters
        ----------
        events
            Buffer of events to write (structured array or EventArray)

        Returns
        -------
        int
            Number of events written
        '''
        n_written = self._file_encoder.write(events)
        self._n_written_events += n_written
        return n_written

    def flush(self):
        '''
        Flush the buffer to the file
        '''
        self._file_encoder.flush()

    def __enter__(self):
        return self

    def __repr__(self) -> str:
        if self._is_initialized:
            is_initialized_txt = f"Written {self._n_written_events} events"
        else:
            is_initialized_txt = "not initialized"
        return f"{self.__class__.__name__}(file={self._file} - {is_initialized_txt}, {self._width}x{self._height})"

    def __len__(self) -> int:
        return self._n_written_events

    def close(self):
        '''
        Close the writer and release the resources
        '''
        self.flush()
        self._file.close()

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
