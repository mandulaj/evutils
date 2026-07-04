
import io
from datetime import datetime
from pathlib import Path
from typing import Union

import numpy as np

from ..io import writer as ev_writers
from ._common import EventEncoder_Base


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
    file_writer
        File writer to use, by default 'auto'
    **kwargs
        Additional arguments for the file writer

    Examples
    --------
    >>> events = np.array([(0, 0, 0, 1), (1000, 10, 10, 0)], dtype=Event_dtype)
    >>> with EventWriter("events.raw", delta_t=10000) as writer:
    >>>     writer.write(events)
    '''
    def __init__(self, file: Path | str | io.BufferedWriter, width:int=1280, height:int=720, dt: datetime|None = None,  file_encoder: EventEncoder_Base | None = None, **kwargs):

        self.file_name:Path|None = None

        # Handle paths as input
        # if file is not a Path, convert it to a Path
        if isinstance(file, str):
            file = Path(file)
        if isinstance(file, Path):

            self.file_name = file

            file = self._open_file(file)

        else:
            # File was passed a io.BufferedReader - we need an explicit file_decoder
            if file_encoder is None:
                raise ValueError(f"When using a io.BufferedWriter as file, the file_decoder must be provided explicitly")

        if isinstance(file, io.BufferedWriter):
            if not file.writable():
                raise IOError("File is not writable")
        self.file: io.BufferedWriter = file

        # File decoder for differnt file types
        if file_encoder is None:
            assert self.file_name is not None
            self.file_encoder = self._create_file_encoder(self.file_name, kwargs)
        else:
            self.file_encoder = file_encoder

        self.width = width
        self.height = height

        self.n_written_events = 0

        self.is_initialized = False

        if dt is None:
            self.dt = datetime.now()
        else:
            self.dt = dt

        # File writer for differnt file types
        if file_encoder is None:
            assert self.file_name is not None
            self.file_writer = self._create_file_encoder(self.file_name, kwargs)
        else:
            self.file_writer = file_encoder

    def _open_file(self, file_name: Path) -> io.BufferedWriter:
        return open(str(file_name), 'wb')


    def _create_file_encoder(self, file_name:Path, args:dict={}) -> EventEncoder_Base:
        '''
        Create the file writer based on the file extension

        Returns
        -------
        EventFileWriter_Base
            The file writer
        '''

        encoder_cls = ev_writers.get_file_writer(file_name)

        return encoder_cls(self.file, **args)

    def init(self):
        '''
        Initialize the writer (e.g. open the file, write the header)

        This method can be called explicitly, but it is also called automatically when the first event is written
        '''
        self.file_writer.init()

    def write(self, events: np.ndarray) -> int:
        '''
        Write a buffer of events to the file

        Parameters
        ----------
        events
            Buffer of events to write

        Returns
        -------
        int
            Number of events written
        '''
        n_written = self.file_writer.write(events)
        self.n_written_events += n_written

        return n_written

        self.file = file
    def flush(self):
        '''
        Flush the buffer to the file
        '''
        self.file_writer.flush()

    def __enter__(self):
        return self


    def __repr__(self) -> str:
        if self.is_initialized:
            is_initialized_txt = f"Written {self.n_written_events} events"
        else:
            is_initialized_txt = "not initialized"
        return f"{self.__class__.__name__}(file={self.file} - {is_initialized_txt}, {self.width}x{self.height})"

    def __len__(self) -> int:
        return self.n_written_events

    def close(self):
        '''
        Close the file reader and release the resources
        '''
        self.flush()
        self.file.close()

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
