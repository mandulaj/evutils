"""Reading and writing events in various file formats.

Format-specific readers and writers (``bin``, ``csv``, ``dat``, ``hdf5``,
``npz``, ``raw``, ``txt``) built on a shared ``EventReader`` / ``EventWriter``
interface, plus ``EventReader_Any`` / ``EventWriter_Any`` which pick the
backend from the file extension. Backends that need optional dependencies
degrade gracefully and only raise on use.
"""

from ._event_reader import EventReader
from ._event_writer import EventWriter
from .stream import EventStreamer

__all__ = [
    "EventReader",
    "EventWriter",
    "EventStreamer",
]