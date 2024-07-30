
from ._writer import EventWriter
from ._reader import EventReader

import numpy as np
import pandas as pd


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
    max_events : int, optional
        Maximum number of events to read at once, by default 10,000
    mode : {"auto", "delta_t", "n_events", "mixed", "all" }, optional
        Mode of operation, by default "auto"
    buffer_size : int, optional
        Number of events to read at once, by default 1,000,000
    order : list, optional
        Order of the columns in the CSV file, by default ['t', 'x', 'y', 'p']
    header : bool, optional
        If True, the first line is considered a header, by default True

    Raises
    ------
    ValueError
        If the order is not a list of 4 strings or if the order does not contain 't', 'x', 'y' and 'p'
    
    '''
    def __init__(self, file,  delta_t=None, n_events=None, max_events=10_000, mode="auto", buffer_size=1_000_000, order=['t', 'x', 'y', 'p'], header=True):
        super().__init__(file, delta_t, n_events, max_events, mode)

        if len(order) != 4:
            raise ValueError("Order must be a list of 4 strings")
        if "t" not in order or "x" not in order or "y" not in order or "p" not in order:
            raise ValueError("Order must contain 't', 'x', 'y' and 'p'")
        
        self.buffer_size = buffer_size
        self.order = order
        self.header = header

        self.chunk_reader = None

    def init(self):
        self.fd = open(self.file, "r")

        # Check first line to see if it is a header
        first_line = self.fd.readline()

        print(first_line)
        self.fd.seek(0)

        self.chunk_reader = pd.read_csv(self.file, chunksize=self.buffer_size, header=None, names=self.order)


        self.is_initialized = True


    def read(self, delta_t=None, n_events=None) -> np.ndarray:
        if not self.is_initialized:
            self.init()


        

        return df.to_numpy()

        if self.mode == "delta_t":
            raise NotImplementedError("delta_t mode not implemented yet")
        elif self.mode == "n_events":
            raise NotImplementedError("n_events mode not implemented yet")
        elif self.mode == "mixed":
            raise NotImplementedError("mixed mode not implemented yet")
        elif self.mode == "all":
            return self.fd.read()
        else:
            raise ValueError(f"Mode {self.mode} not supported")




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