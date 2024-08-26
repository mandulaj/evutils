


from io import TextIOWrapper
from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd
from pandas.io.parsers import TextFileReader

from ..types import Event_dtype
from ._common import EventFileReader, EventFileWriter


class EventFileReader_Csv(EventFileReader):
    '''
    A reader for CSV files with events.

    Parameters
    ----------
    file
        Path to the data file
    order
        Order of the columns in the CSV file, by default ['t', 'x', 'y', 'p']
    header
        If True, the first line is considered a header, by default True
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

    def __init__(self, file: Path,  order:list=['t', 'x', 'y', 'p'], header:bool=True, chunk_size:int=1_000_000, delimiter:str=",", engine:str='c'):
        super().__init__(file, chunk_size)

        # Validate the parameters
        if len(order) != 4:
            raise ValueError("Order must be a list of 4 strings")
        if "t" not in order or "x" not in order or "y" not in order or "p" not in order:
            raise ValueError("Order must contain 't', 'x', 'y' and 'p'")

        self.order = order
        self.header = header
        self.delimiter = delimiter
        self.engine = engine

        self.chunk_reader:TextFileReader|None= None

    def _skip_header(self):

        assert self.fd is not None
        # Check first line to see if it is a header
        first_line: str = self.fd.readline()

        # Check if the first line is a header
        # If we find a header, it will take precendence over the order parameter

        if "t" in first_line or "x" in first_line or "y" in first_line or "p" in first_line:
            # Header found
            self.header = True

        if self.header:
            # We expect a header:
            if "t" in first_line and "x" in first_line and "y" in first_line and "p" in first_line:
                # Header found
                order = first_line.strip().split(",")
                if order != self.order:
                    print(ValueError(f"WARNING: Header order {self.order} does not match order file {order}"))
                    self.order = order


            else:
                raise ValueError(f"Header not found or invalid: {first_line}")
        else:
            # No header found, just seek to start of file
            self.fd.seek(0)



    def init(self):
        self.fd = open(self.file, "r")

        self._skip_header()

        self.chunk_reader = pd.read_csv(self.fd, iterator=True, header=None, names=self.order, engine=self.engine,
                                        delimiter=self.delimiter,  dtype={"t":"u8", "p":"u1", "x":"u2","y":"u2"})

        # self.numpy_reader = np.fromtxt(self.fd, delimiter=",", dtype=np.uint64)
        self.is_initialized = True


    def read_chunk(self, delta_t_hint:int = None, n_events_hint:int = None) -> np.ndarray:
        assert self.is_initialized, "Reader is not initialized"
        assert self.chunk_reader is not None
        # We can use the n_events_hint to read exactly n_events
        #n_events_hint = None
        #if not n_events_hint is None:
        #    chunk_size = n_events_hint
        #else:
        #    chunk_size = self.chunk_size
        chunk_size = self.chunk_size

        try:
            buffer = self.chunk_reader.get_chunk(chunk_size)

            buffer = np.array(buffer[['t', 'x', 'y', 'p']].to_records(index=False), dtype=Event_dtype)
        except StopIteration:
            buffer = np.array([], dtype=Event_dtype)
            self.eof = True

        return buffer


    def reset(self):
        assert self.fd is not None
        self.fd.seek(0)
        self._skip_header()

class EventFileWriter_Csv(EventFileWriter):
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

    def __init__(self, file: Path, width:int=1280, height:int=720, sep:str=",", order:list=['t', 'x', 'y', 'p'], header:bool=True):
        super().__init__(file)


        if len(order) != 4:
            raise ValueError("Order must be a list of 4 strings")
        if "t" not in order or "x" not in order or "y" not in order or "p" not in order:
            raise ValueError("Order must contain 't', 'x', 'y' and 'p'")

        self.order = order
        self.header = header
        self.sep = sep

    def init(self):
        self.fd = open(self.file, "w")

        if self.header:
            self.fd.write(self.sep.join(self.order) + "\n")

        self.is_initialized = True


    def write(self, events: np.ndarray):
        if not self.is_initialized:
            self.init()

        df = pd.DataFrame(events)
        df.to_csv(self.fd, header=False, index=False, columns=self.order, sep=self.sep)

    def close(self):
        if self.is_initialized:
            assert self.fd is not None
            self.fd.close()


