"""Transparent (de)compression helpers for path-based IO.

Event files are frequently stored compressed (``foo.raw.zst``,
``events.csv.gz``). This module centralises the mapping from a compression
suffix to the right stdlib (or third-party) file wrapper so both the reader
(:mod:`evutils.io._source`) and the writer
(:class:`~evutils.io._event_writer.EventWriter`) can open them transparently.

The *inner* extension still selects the event format: ``foo.raw.zst`` is an
EVT file read/written through a zstd stream. Use
:func:`strip_compression_suffix` to recover the inner name for format
detection.

Supported suffixes:

* ``.gz``  -- gzip     (stdlib :mod:`gzip`)
* ``.xz``  -- lzma/xz  (stdlib :mod:`lzma`)
* ``.bz2`` -- bzip2    (stdlib :mod:`bz2`)
* ``.zst`` -- zstandard: stdlib ``compression.zstd`` (Python 3.14+), falling
  back to the third-party ``zstandard`` or ``pyzstd`` packages if installed.
"""
from __future__ import annotations

import io
from pathlib import Path

#: Recognised compression suffixes (lower-case, incl. leading dot).
COMPRESSION_SUFFIXES = {".gz", ".zst", ".xz", ".bz2"}

def is_compressed_path(path: "str | Path") -> bool:
    """Return True if ``path``'s final suffix is a known compression suffix."""
    return Path(path).suffix.lower() in COMPRESSION_SUFFIXES

def strip_compression_suffix(name: str) -> str:
    """Drop a trailing compression suffix: ``'foo.raw.zst'`` -> ``'foo.raw'``.

    Names without a compression suffix are returned unchanged.
    """
    p = Path(name)
    suffix = p.suffix
    if suffix.lower() in COMPRESSION_SUFFIXES:
        return name[:-len(suffix)]
    return name

def _open_zstd(path: "str | Path", mode: str) -> "io.BufferedIOBase":
    """Open a ``.zst`` file, trying stdlib then third-party backends."""
    try:
        from compression.zstd import ZstdFile  # Python 3.14+ stdlib
        return ZstdFile(path, mode)
    except ImportError:
        pass
    try:
        import zstandard  # third-party
        return zstandard.open(path, mode)
    except ImportError:
        pass
    try:
        import pyzstd  # third-party
        return pyzstd.ZstdFile(path, mode)
    except ImportError:
        pass
    raise ImportError(
        "reading/writing '.zst' files requires zstd support: use Python 3.14+ "
        "(stdlib 'compression.zstd') or install the 'zstandard' or 'pyzstd' "
        "package"
    )

def open_compressed(path: "str | Path", mode: str = "rb") -> "io.BufferedIOBase":
    """Open a compressed file, dispatching on its suffix.

    Parameters
    ----------
    path
        Path whose final suffix is one of :data:`COMPRESSION_SUFFIXES`.
    mode
        Binary open mode (``'rb'`` / ``'wb'``). Text modes are not supported --
        event codecs operate on bytes.

    Returns
    -------
    io.BufferedIOBase
        A binary, decompressing/compressing file object wrapping ``path``.

    Raises
    ------
    ValueError
        If the suffix is not a recognised compression suffix.
    ImportError
        If the backend for the suffix is unavailable (e.g. zstd).

    """
    suffix = Path(path).suffix.lower()
    if suffix == ".gz":
        import gzip
        return gzip.open(path, mode)
    if suffix == ".xz":
        import lzma
        return lzma.open(path, mode)
    if suffix == ".bz2":
        import bz2
        return bz2.open(path, mode)
    if suffix == ".zst":
        return _open_zstd(path, mode)
    raise ValueError(
        f"{suffix!r} is not a supported compression suffix "
        f"(expected one of {sorted(COMPRESSION_SUFFIXES)})"
    )
