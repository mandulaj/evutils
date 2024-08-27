


import numpy as np

from ._common import EventDecoder, EventEncoder


class EventFileReader_Txt(EventDecoder):
    def __init__(self, file):
        super().__init__(file)


class EventFileWriter_Txt(EventEncoder):
    def __init__(self, file, width=1280, height=720):
        super().__init__(file, width, height)
