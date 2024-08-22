


from ._common import EventFileReader, EventFileWriter

import numpy as np


class EventFileReader_Aedat(EventFileReader):
    def __init__(self, file):
        super().__init__(file)


class EventFileWriter_Aedat(EventFileWriter):
    def __init__(self, file, width=1280, height=720):
        super().__init__(file)

