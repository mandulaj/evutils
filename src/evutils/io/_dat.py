
import numpy as np

from .common import EventDecoder, EventEncoder


class EventDecoder_Dat(EventDecoder):
    '''
    A decoder for reading events from DAT files.

    Parameters
    ----------
    file
        Path to the data file
    '''
    def __init__(self, file):
        super().__init__(file)


class EventEncoder_Dat(EventEncoder):
    '''
    A encoder for writing events to DAT files.

    Parameters
    ----------
    file
        Path to the data file
    '''
    def __init__(self, file, width=1280, height=720):
        super().__init__(file)
