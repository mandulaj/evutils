
from ._writer import EventWriter
from ._reader import EventReader

import numpy as np
import pandas as pd


class EventReader_Csv(EventReader):
    def __init__(self, file):
        super().__init__(file)



class EventWriter_Csv(EventWriter):
    def __init__(self, file, width=1280, height=720, order=['t', 'x', 'y', 'p'], header=True):
        super().__init__(file, width, height)

        if len(order) != 4:
            raise ValueError("Order must be a list of 4 strings")
        if "t" not in order or "x" not in order or "y" not in order or "p" not in order:
            raise ValueError("Order must contain 't', 'x', 'y' and 'p'")
        
        self.order = order
        self.header = header

    def init(self):
        self.fd = open(self.file, "w")

        if self.header:
            self.fd.write(",".join(self.order) + "\n")

        self.is_initialized = True


    def write(self, events: np.ndarray):
        if not self.is_initialized:
            self.init()
            
        df = pd.DataFrame(events)
        df.to_csv(self.fd, header=False, index=False, columns=self.order)

    def close(self):
        if self.is_initialized:
            self.fd.close()