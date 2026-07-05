


import io
from io import TextIOWrapper
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pandas._typing import CSVEngine
from pandas.io.parsers import TextFileReader

from ..types import Event_dtype, EventArray
from .common import EventDecoder, EventEncoder


class EventDecoder_Csv(EventDecoder):
    '''
    A reader for CSV files with events.

    Parameters
    ----------
    file
        Path to the data file
    order
        Order of the columns in the CSV file, by default ['t', 'x', 'y', 'p']
    chunk_size
        Size of the chunk to read, by default 10_000
    delimiter
        Delimiter for the CSV file, by default ","
    engine
        Engine to use to read the CSV file, by default 'c'

    Raises
    ------
    ValueError
        If the order is not a list of 4 strings or if the order does not contain 't', 'x', 'y' and 'p'

    '''

    def __init__(self, readable:io.BufferedReader,  order:list|None=None, chunk_size:int=1_000_000, delimiter:str=",", engine:CSVEngine='c'):
        super().__init__(readable, chunk_size)


        if order is None:
            # we infer the header from the file, or use a default header
            pass
        else:
            # Validate the parameters
            if len(order) != 4:
                raise ValueError("Order must be a list of 4 strings")
            if "t" not in order or "x" not in order or "y" not in order or "p" not in order:
                raise ValueError("Order must contain 't', 'x', 'y' and 'p'")

        self._order = order
        self._delimiter = delimiter
        self._engine = engine

        self._chunk_reader:TextFileReader|None= None

    def _check_header(self):
        have_header = False

        # Check first line to see if it is a header
        # TODO: Need to deal with non-seekable files
        first_line: str = self._fd.readline().decode('utf-8').strip()

        # Check if the first line is a header
        # If we find a header, it will take precendence over the order parameter

        if "t" in first_line or "x" in first_line or "y" in first_line or "p" in first_line:
            # Header found
            have_header = True

        if have_header:
            # We expect a header:
            if "t" in first_line and "x" in first_line and "y" in first_line and "p" in first_line:
                # Header found
                order = first_line.split(",")

                if self._order is not None and order != self._order:
                    print(ValueError(f"WARNING: Header order {self._order} does not match order file {order}"))
                self._order = order


            else:
                raise ValueError(f"Header not found or invalid: {first_line}")
        else:
            # No header found, just seek to start of file
            self._fd.seek(0)

        # If we still don't have a header, we use a default one
        if self._order is None:
            self._order = ['t', 'x', 'y', 'p']



    def init(self):

        self._check_header()

        self._chunk_reader = pd.read_csv(self._fd, iterator=True, header=None, names=self._order, engine=self._engine,
                                        delimiter=self._delimiter,  dtype={"t":"u8", "p":"u1", "x":"u2","y":"u2"})

        # self.numpy_reader = np.fromtxt(self._fd, delimiter=",", dtype=np.uint64)
        self._is_initialized = True


    def read_chunk(self, delta_t_hint:int | None = None, n_events_hint:int | None = None) -> np.ndarray[Any, np.dtype[Any]]:
        assert self._is_initialized, "Reader is not initialized"
        assert self._chunk_reader is not None
        # We can use the n_events_hint to read exactly n_events
        #n_events_hint = None
        #if not n_events_hint is None:
        #    chunk_size = n_events_hint
        #else:
        #    chunk_size = self.chunk_size
        chunk_size = self._chunk_size

        try:
            df = self._chunk_reader.get_chunk(chunk_size)
            events = EventArray(df['t'].to_numpy(), df['x'].to_numpy(),
                                df['y'].to_numpy(), df['p'].to_numpy())
        except StopIteration:
            events = EventArray.empty()

        # Detect end of file
        if len(events) < chunk_size:
            self._eof = True

        return events


    def reset(self):
        assert self._fd is not None
        self._fd.seek(0)
        self._check_header()



class EventEncoder_Csv(EventEncoder):
    '''
    A writer for CSV files with events.

    Parameters
    ----------
    file : str
        Path to the data file
    width : int, optional
        Width of the frame, by default 1280 (not relevant for this formats)
    height : int, optional
        Height of the frame, by default 720 (not relevant for this formats)
    sep : str, optional
        Separator for the CSV file, by default ","
    order : list, optional
        Order of the columns in the CSV file, by default ['t', 'x', 'y', 'p']
    header : bool, optional
        If True, a header is written on the first line, by default True

    Raises
    ------
    ValueError
        If the order is not a list of 4 strings or if the order does not contain 't', 'x', 'y' and 'p'

    '''

    def __init__(self, writable: io.BufferedWriter, width:int=1280, height:int=720, sep:str=",", order:list=['t', 'x', 'y', 'p'], header:bool=True):
        super().__init__(writable)


        if len(order) != 4:
            raise ValueError("Order must be a list of 4 strings")
        if "t" not in order or "x" not in order or "y" not in order or "p" not in order:
            raise ValueError("Order must contain 't', 'x', 'y' and 'p'")

        self._order = order
        self._header = header
        self._sep = sep

    def init(self):

        if self._header:
            header = self._sep.join(self._order) + "\n"
            self._fd.write(header.encode('utf-8'))

        self._is_initialized = True


    def write(self, events: np.ndarray[Any, np.dtype[Any]]) -> int:
        if not self._is_initialized:
            self.init()

        df = pd.DataFrame(events)
        df.to_csv(self._fd, header=False, index=False, columns=self._order, sep=self._sep)
        return len(events)

