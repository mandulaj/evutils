"""Binary (.bin) file decoder and encoder.

.. note::
    Not implemented yet: both classes raise :class:`NotImplementedError` on
    construction. The classes exist so the ``.bin`` extension is reserved in
    the reader/writer registries and the error message is explicit.
"""

import numpy as np
from ..types import EventArray, TriggerArray
from .common import EventDecoder, EventEncoder

_NOT_IMPLEMENTED = (
    "The .bin event format is not implemented yet. "
    "Convert the file to a supported format (RAW/EVT, DAT, AER, HDF5, NPZ, CSV) "
    "or pass an explicit file_decoder/file_encoder."
)

class EventDecoder_Bin(EventDecoder):
    """A decoder for reading events from binary files.

    .. note::
        Not implemented yet -- constructing this class raises
        :class:`NotImplementedError`.

    Parameters
    ----------
    source : ByteSource
        Byte source to read events from.
    **kwargs
        Additional decoder arguments.

    Raises
    ------
    NotImplementedError
        Always, until the format is implemented.

    """

    def __init__(self, source: "io.BufferedIOBase | str | bytes", **kwargs):
        raise NotImplementedError(_NOT_IMPLEMENTED)

    def init(self) -> None:
        """Initialize the file for reading."""
        raise NotImplementedError(_NOT_IMPLEMENTED)

    def read_chunk(self, delta_t_hint: int | None = None, n_events_hint: int | None = None) -> "np.ndarray | EventArray | None":
        """Read a chunk of events."""
        raise NotImplementedError(_NOT_IMPLEMENTED)

    def reset(self) -> None:
        """Reset the file pointer to the beginning of the file."""
        raise NotImplementedError(_NOT_IMPLEMENTED)

class EventEncoder_Bin(EventEncoder):
    """An encoder for writing events to binary files.

    .. note::
        Not implemented yet -- constructing this class raises
        :class:`NotImplementedError`.

    Parameters
    ----------
    writable : io.BufferedIOBase
        Destination for writing events.
    **kwargs
        Additional encoder arguments (``width``, ``height``, ``dt``, ...).

    Raises
    ------
    NotImplementedError
        Always, until the format is implemented.

    """

    def __init__(self, writable: "io.BufferedIOBase", **kwargs):
        raise NotImplementedError(_NOT_IMPLEMENTED)

    def init(self) -> None:
        """Initialize the file for writing."""
        raise NotImplementedError(_NOT_IMPLEMENTED)

    def write(self, events: 'np.ndarray | EventArray', triggers: 'np.ndarray | TriggerArray | None' = None) -> int:
        """Write a chunk of events."""
        raise NotImplementedError(_NOT_IMPLEMENTED)
