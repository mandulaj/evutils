"""Binary (.bin) file decoder and encoder."""

import numpy as np

from .common import EventDecoder, EventEncoder


class EventDecoder_Bin(EventDecoder):
    """A decoder for reading events from binary files.

    Parameters
    ----------
    file : str or Path
        Path to the data file.

    """

    def __init__(self, file):
        super().__init__(file)


class EventEncoder_Bin(EventEncoder):
    """An encoder for writing events to binary files.

    Parameters
    ----------
    file : str or Path
        Path to the data file.
    width : int, optional
        Width of the frame, by default 1280.
    height : int, optional
        Height of the frame, by default 720.

    """

    def __init__(self, file, width=1280, height=720):
        super().__init__(file)

