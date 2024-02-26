
from ._writer import EventWriter
from ._reader import EventReader

import numpy as np


class EventReader_Csv(EventReader):
    def __init__(self, file):
        super().__init__(file)



class EventWriter_Csv(EventWriter):
    def __init__(self, file, width=1280, height=720):
        super().__init__(file, width, height)

    def init(self):
        self.fd = open(self.file, "w")
        self.fd.write("t, x, y, p\n")

    def write(self, events: np.ndarray):
        for ev in events:
            self.fd.write(f"{ev['t']}, {ev['x']}, {ev['y']}, {ev['p']}\n")

    def close(self):
        self.fd.close()