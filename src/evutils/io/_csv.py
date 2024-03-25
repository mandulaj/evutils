
from ._writer import EventWriter
from ._reader import EventReader

import numpy as np
import pandas as pd


class EventReader_Csv(EventReader):
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