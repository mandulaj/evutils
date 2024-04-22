
from ._writer import EventWriter
from ._reader import EventReader

import numpy as np
import pandas as pd

from ..types import Events


class EventReader_Csv(EventReader):
    def __init__(self, file,  delta_t=None, n_events=None,  mode="auto", start_ts=0,  max_time=1_000_000_000_000, max_events=10_000_000, width=None, height=None, order=['t', 'x', 'y', 'p'], header=True):
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