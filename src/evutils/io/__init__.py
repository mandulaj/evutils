"""Reading and writing events in various file formats.

Format-specific readers and writers (``bin``, ``csv``, ``dat``, ``hdf5``,
``npz``, ``raw``, ``txt``) built on a shared ``EventReader`` / ``EventWriter``
interface, plus ``EventReader_Any`` / ``EventWriter_Any`` which pick the
backend from the file extension. Backends that need optional dependencies
degrade gracefully and only raise on use.

Compression is transparent: a path ending in ``.gz`` / ``.zst`` / ``.xz`` /
``.bz2`` (e.g. ``foo.raw.zst``) is auto-opened through the matching
(de)compressing stream on both read and write -- the *inner* extension selects
the event format. You may also hand :class:`EventReader` an already-open
compressed file object (``gzip.GzipFile``, ``lzma.LZMAFile``,
``compression.zstd.ZstdFile``, ...). Writing to a bare stream (as opposed to a
path) still requires an explicit ``file_encoder``. ``.zst`` needs Python 3.14+
(stdlib ``compression.zstd``) or the ``zstandard`` / ``pyzstd`` package.
"""

from ._event_reader import EventReader
from ._event_writer import EventWriter
from .stream import EventStreamer

__all__ = [
    "EventReader",
    "EventWriter",
    "EventStreamer",
]