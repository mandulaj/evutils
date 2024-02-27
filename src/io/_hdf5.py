
from ._writer import EventWriter
from ._reader import EventReader

import numpy as np


class EventReader_HDF5(EventReader):
    def __init__(self, file):
        super().__init__(file)


class EventWriter_HDF5(EventWriter):
    def __init__(self, file, width=1280, height=720):
        super().__init__(file, width, height)