"""NPZ file decoder and encoder."""

import numpy as np

from .common import EventDecoder, EventEncoder


class EventDecoder_Npz(EventDecoder):
    """A decoder for reading events from NPZ files.

    Parameters
    ----------
    file : str or Path
        Path to the data file.

    """

    def __init__(self, file):
        super().__init__(file)


class EventEncoder_Npz(EventEncoder):
    """An encoder for writing events to NPZ files.

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
