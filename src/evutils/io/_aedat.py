
"""AEDAT file decoder and encoder."""

import numpy as np

from .common import EventDecoder, EventEncoder


class EventDecoder_Aedat(EventDecoder):
    """A decoder for reading events from AEDAT files.

    Parameters
    ----------
    file : str or Path
        Path to the data file.

    """

    def __init__(self, file):
        super().__init__(file)


class EventEncoder_Aedat(EventEncoder):
    """An encoder for writing events to AEDAT files.

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

