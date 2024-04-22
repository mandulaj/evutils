
from ._writer import EventWriter
from ._reader import EventReader

import numpy as np
import pandas as pd

from ..types import Events


class EventReader_Csv(EventReader):
    '''
    A reader for CSV files with events.

    Parameters
    ----------
    file : str
        Path to the data file
    delta_t : int, optional
        Time interval between events in microseconds, by default None
    n_events : int, optional
        Number of events to read in a chunk, by default None
    mode : {"auto", "delta_t", "n_events", "mixed", "all" }, optional
        Mode of operation, by default "auto"
    start_ts : int, optional
        Start timestamp for the events, by default 0
    max_time : int, optional
        Maximum timestamp to read, by default 1_000_000_000_000
    max_events : int, optional
        Maximum number of events to read at once, by default 10,000
    width : int, optional
        Width of the frame, by default None
    height : int, optional
        Height of the frame, by default None
    order : list, optional
        Order of the columns in the CSV file, by default ['t', 'x', 'y', 'p']
    header : bool, optional
        If True, the first line is considered a header, by default True

    Raises
    ------
    ValueError
        If the order is not a list of 4 strings or if the order does not contain 't', 'x', 'y' and 'p'
    
    '''

    def __init__(self, file,  delta_t:int=None, n_events:int=None,  mode:str="auto", start_ts:int=0,  max_time:int=1_000_000_000_000, max_events:int=10_000_000, width:int=None, height:int=None, order=['t', 'x', 'y', 'p'], header:bool=True):
        super().__init__(file=file, delta_t=delta_t, n_events=n_events, mode=mode, start_ts=start_ts, max_time=max_time, max_events=max_events, width=width, height=height)
        
        # Validate the parameters
        if len(order) != 4:
            raise ValueError("Order must be a list of 4 strings")
        if "t" not in order or "x" not in order or "y" not in order or "p" not in order:
            raise ValueError("Order must contain 't', 'x', 'y' and 'p'")
        
        self.order = order
        self.header = header

        self.chunk_reader = None

    def init(self):
        self.fd = open(self.file, "r")

        # Check first line to see if it is a header
        first_line = self.fd.readline()

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



        self.chunk_reader = pd.read_csv(self.fd, iterator=True, header=None, names=self.order, engine='c',  dtype={"t":"u8", "p":"u1", "x":"u2","y":"u2"})

        # self.numpy_reader = np.fromtxt(self.fd, delimiter=",", dtype=np.uint64)
        self.is_initialized = True


    def _read(self, delta_t, n_events) -> np.ndarray:
        # print(f"Reading from csv {delta_t}, {n_events}")

        buffer = self.chunk_reader.get_chunk(n_events)
        
        buffer = np.array(buffer.to_records(index=False), dtype=Events)

        if len(buffer) == 0:
            self.eof = True
        return buffer





class EventWriter_Csv(EventWriter):
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
    
    
    
    def __init__(self, file, width=1280, height=720, sep=",", order=['t', 'x', 'y', 'p'], header=True):
        super().__init__(file, width, height)

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
            self.fd.close()