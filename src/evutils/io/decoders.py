"""Decoders module.

Provides mapping and resolution of event decoders based on file extensions or
magic bytes. Backends whose optional dependencies are missing (e.g. pandas for
CSV, h5py for HDF5) are not registered; asking for them raises an
``ImportError`` that names the extra to install.
"""

from pathlib import Path
from typing import Type, Any, cast

from .common import EventDecoder

#: Extension -> decoder class, for available backends only.
_READER_MAPPING: dict[str, Type[EventDecoder]] = {}

#: Extension -> reason it is unavailable (missing optional dependency).
_UNAVAILABLE: dict[str, str] = {}


from ._aedat import EventDecoder_Aedat
_READER_MAPPING[".aedat"] = EventDecoder_Aedat
_READER_MAPPING[".aedat4"] = EventDecoder_Aedat

from ._bin import EventDecoder_Bin
_READER_MAPPING[".bin"] = EventDecoder_Bin

try:
    from ._csv import EventDecoder_Csv
    _READER_MAPPING[".csv"] = EventDecoder_Csv
    _READER_MAPPING[".txt"] = EventDecoder_Csv
except ImportError:
    _UNAVAILABLE[".csv"] = _UNAVAILABLE[".txt"] = (
        "reading CSV/TXT event files requires the evutils native library "
        "(build it with `uv pip install -e .`)"
    )

from ._dat import EventDecoder_Dat
_READER_MAPPING[".dat"] = EventDecoder_Dat

try:
    from ._hdf5 import EventDecoder_HDF5
    _READER_MAPPING[".hdf5"] = EventDecoder_HDF5
    _READER_MAPPING[".h5"] = EventDecoder_HDF5
except ImportError:
    _UNAVAILABLE[".hdf5"] = _UNAVAILABLE[".h5"] = (
        "reading HDF5 event files requires h5py/hdf5plugin: install `evutils[hdf5]`"
    )

from ._npz import EventDecoder_Npz
_READER_MAPPING[".npz"] = EventDecoder_Npz

from ._evt import EventDecoder_EVT
_READER_MAPPING[".raw"] = EventDecoder_EVT
_READER_MAPPING[".evt"] = EventDecoder_EVT
_READER_MAPPING[".evt3"] = EventDecoder_EVT
_READER_MAPPING[".evt2"] = EventDecoder_EVT
_READER_MAPPING[".evt21"] = EventDecoder_EVT

from ._aer import EventDecoder_AER
_READER_MAPPING[".aer"] = EventDecoder_AER


def _lookup(ext: str) -> Type[EventDecoder]:
    """Resolve an extension to a decoder class, or raise a helpful error."""
    if ext in _READER_MAPPING:
        return _READER_MAPPING[ext]
    if ext in _UNAVAILABLE:
        raise ImportError(f"File extension {ext} is supported, but {_UNAVAILABLE[ext]}")
    raise ValueError(
        f"File extension {ext} not supported, available extensions: "
        f"{sorted(_READER_MAPPING.keys() | _UNAVAILABLE.keys())}"
    )


def get_reader_from_filename(file: Path) -> Type[EventDecoder]:
    """Get the appropriate reader for the given file.

    Parameters
    ----------
    file
        File to read

    Returns
    -------
    EventDecoder
        Reader object for the file

    """
    return _lookup(file.suffix.lower())


# Content sniffers: (predicate over the first bytes -> decoder class). Tried in
# order when the filename extension is unknown or absent (streams, USB).
def _header_lines(head: bytes) -> list[str]:
    """The ``"% ..."`` ASCII header lines at the start of ``head``."""
    text = head.decode("ascii", "ignore")
    return [ln for ln in text.split("\n") if ln.startswith("% ")]


def _sniff_dat(head: bytes) -> bool:
    """Prophesee DAT: header carries ``% Version`` / ``% Data file containing``.

    DAT and RAW/EVT both open with a ``%`` header, so they can only be told
    apart by the keywords inside it.
    """
    for ln in _header_lines(head):
        low = ln.lower()
        if "data file containing" in low or low.startswith("% version"):
            return True
    return False


def _sniff_evt(head: bytes) -> bool:
    """Prophesee RAW/EVT: header carries ``% evt`` / ``% format EVT`` / ``% geometry``."""
    for ln in _header_lines(head):
        low = ln.lower()
        if (low.startswith("% evt")
                or low.startswith("% format evt")
                or low.startswith("% geometry")):
            return True
    return False


def _sniff_prophesee(head: bytes) -> bool:
    """Fallback: any other ``"% "``-headed stream is treated as RAW/EVT."""
    return head[:2] == b"% "


def _sniff_aedat(head: bytes) -> bool:
    """Check if the first bytes match an AEDAT version line.

    Parameters
    ----------
    head : bytes
        The first bytes of the file/stream.

    Returns
    -------
    bool
        True if it matches the AEDAT format, False otherwise.

    """
    return head.startswith(b"#!AER-DAT")


_SNIFFERS = [
    (_sniff_dat, "EventDecoder_Dat"),
    (_sniff_evt, "EventDecoder_EVT"),
    (_sniff_aedat, "EventDecoder_Aedat"),
    (_sniff_prophesee, "EventDecoder_EVT"),  # generic "% "-headed fallback
]


def resolve_decoder_cls(source: Any) -> Type[EventDecoder]:
    """Determine the decoder class for a :class:`ByteSource`.

    Tries the filename extension first (cheap, usually right), then falls back
    to sniffing the leading bytes -- which works for streams and USB devices
    that have no filename.

    Parameters
    ----------
    source
        A ByteSource (see :mod:`evutils.io._source`).

    Returns
    -------
    Type[EventDecoder]
        The decoder class to instantiate with the source.

    """
    name = getattr(source, "name", None)
    if name:
        ext = Path(name).suffix.lower()
        if ext in _READER_MAPPING or ext in _UNAVAILABLE:
            return _lookup(ext)

    try:
        head = source.peek(512)
    except Exception:
        head = b""

    for matches, cls_name in _SNIFFERS:
        if matches(head):
            return cast(Type[EventDecoder], globals()[cls_name])

    raise ValueError(
        "Could not determine the event format: unknown/absent extension "
        f"({name!r}) and no known magic bytes. Pass an explicit decoder."
    )


__all__ = ["EventDecoder", 'EventDecoder_Aedat', 'EventDecoder_Bin', 'EventDecoder_Csv', 'EventDecoder_Dat', 'EventDecoder_HDF5', 'EventDecoder_Npz', 'EventDecoder_EVT', 'EventDecoder_AER', 'get_reader_from_filename', 'resolve_decoder_cls']
