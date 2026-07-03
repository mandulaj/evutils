"""Reading and writing events in various file formats.

Format-specific readers and writers (``bin``, ``csv``, ``dat``, ``hdf5``,
``npz``, ``raw``, ``txt``) built on a shared ``EventReader`` / ``EventWriter``
interface, plus ``EventReader_Any`` / ``EventWriter_Any`` which pick the
backend from the file extension. Backends that need optional dependencies
degrade gracefully and only raise on use.
"""

from ._reader import EventReader
from ._writer import EventWriter


__all__ = ["EventReader", "EventWriter"]