
from ._common import EventFileReader, EventFileWriter


import numpy as np


class EventFileReader_Dat(EventFileReader):
    def __init__(self, file):
        super().__init__(file)


class EventFileWriter_Dat(EventFileWriter):
    def __init__(self, file, width=1280, height=720):
        super().__init__(file)