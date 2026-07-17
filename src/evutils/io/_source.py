"""Byte-level input sources for event decoders.

A :class:`ByteSource` abstracts *where* raw bytes come from (a file, an
in-memory buffer, a memory-mapped file, or -- in the future -- a live USB
device) from *how* they are parsed into events (the decoder). Decoders never
open files or ``seek`` raw handles; they only talk to a ByteSource.

Two capabilities matter:

* Every source supports sequential :meth:`ByteSource.read` -- the lowest common
  denominator, and the only thing a streaming device (USB, pipe) can do.
* Some sources are *mappable*: :meth:`ByteSource.buffer` hands out a zero-copy
  ``memoryview`` of the whole content (mmap, in-memory bytes). Decoders built on
  the native parser prefer this -- the C parser walks the bytes in place with no
  copy and no chunk-boundary carry logic.

Lifetime note: a zero-copy ``buffer()`` (and any ``np.frombuffer`` view of it)
aliases the underlying storage. Drop those views *before* calling
:meth:`ByteSource.close`, or closing an mmap will raise ``BufferError``.
"""
from __future__ import annotations

import io
import mmap
from abc import ABC, abstractmethod
from pathlib import Path

from ._compression import is_compressed_path, open_compressed

class ByteSource(ABC):
    """Abstract raw-byte input. Knows nothing about events."""

    #: Filename if the source has one -- used for extension-based dispatch.
    name: str | None = None

    @abstractmethod
    def read(self, size: int = -1) -> bytes:
        """Read up to ``size`` bytes (all remaining if ``size < 0``).

        Returns ``b""`` at EOF.
        """

    @abstractmethod
    def peek(self, size: int) -> bytes:
        """Return up to ``size`` upcoming bytes *without* consuming them.

        Used for header/magic sniffing. Should work even on non-seekable
        sources; raises :class:`io.UnsupportedOperation` when it cannot.
        """

    def readline(self) -> bytes:
        """Read one line, up to and including the newline (generic, byte-wise).

        Used by text decoders (csv/txt) for header handling. Subclasses backed
        by a real stream override this with the stream's own ``readline``.
        """
        out = bytearray()
        while True:
            b = self.read(1)
            if not b:
                break
            out += b
            if b == b"\n":
                break
        return bytes(out)

    # -- optional zero-copy capability ------------------------------------- #
    def mappable(self) -> bool:
        """Check if the source is mappable.

        Returns
        -------
        bool
            True if mappable, False otherwise.

        """
        return False

    def buffer(self) -> memoryview:
        """Zero-copy view of the *entire* content. Only if :meth:`mappable`."""
        raise io.UnsupportedOperation("source is not mappable")

    # -- optional random access -------------------------------------------- #
    def seekable(self) -> bool:
        """Check if the source is seekable.

        Returns
        -------
        bool
            True if seekable, False otherwise.

        """
        return False

    def seek(self, pos: int, whence: int = io.SEEK_SET) -> int:
        """Seek to a specific position.

        Parameters
        ----------
        pos : int
            Position to seek to.
        whence : int, optional
            Reference point for seeking, by default io.SEEK_SET.

        Returns
        -------
        int
            New position.

        """
        raise io.UnsupportedOperation("source is not seekable")

    def tell(self) -> int:
        """Get the current position.

        Returns
        -------
        int
            Current position.

        """
        raise io.UnsupportedOperation("source is not tellable")

    def reset(self) -> None:
        """Return to the beginning of the stream."""
        self.seek(0)

    def close(self) -> None:
        """Close the source.

        Returns
        -------
        None

        """
        pass

    def __enter__(self) -> "ByteSource":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

def _clamp_seek(pos: int, whence: int, cur: int, length: int) -> int:
    if whence == io.SEEK_SET:
        new = pos
    elif whence == io.SEEK_CUR:
        new = cur + pos
    elif whence == io.SEEK_END:
        new = length + pos
    else:
        raise ValueError(f"invalid whence {whence}")
    return max(0, min(new, length))

class StreamSource(ByteSource):
    """Wrap any binary stream exposing ``read`` (BufferedReader, pipe, ...).

    This is the streaming fallback: it never claims to be mappable, so decoders
    read from it sequentially.
    """

    def __init__(self, stream: "io.BufferedIOBase", name: str | None = None, owns: bool = False) -> None:
        if not hasattr(stream, "read"):
            raise TypeError("stream must have a read() method")
        self._s = stream
        n = name if name is not None else getattr(stream, "name", None)
        self.name = n if isinstance(n, str) else None
        self._owns = owns

    def read(self, size: int = -1) -> bytes:
        return bytes(self._s.read(size))

    def readline(self) -> bytes:
        if hasattr(self._s, "readline"):
            return bytes(self._s.readline())
        return super().readline()

    def peek(self, size: int) -> bytes:
        s = self._s
        if hasattr(s, "peek"):
            return bytes(s.peek(size))[:size]
        if s.seekable():
            pos = s.tell()
            try:
                return bytes(s.read(size))
            finally:
                s.seek(pos)
        raise io.UnsupportedOperation(
            "source is not peekable; cannot sniff format -- pass an explicit decoder"
        )

    def seekable(self) -> bool:
        return bool(self._s.seekable())

    def seek(self, pos: int, whence: int = io.SEEK_SET) -> int:
        return int(self._s.seek(pos, whence))

    def tell(self) -> int:
        return int(self._s.tell())

    def close(self) -> None:
        if self._owns:
            self._s.close()

class BufferSource(ByteSource):
    """Zero-copy source over an in-memory buffer (bytes / bytearray / memoryview
    / ``BytesIO.getbuffer()``).
    """

    def __init__(self, data: bytes, name: str | None = None) -> None:
        self._mv = memoryview(data).cast("B")  # 1-D uint8 view, no copy
        self.name = name
        self._pos = 0

    def read(self, size: int = -1) -> bytes:
        if size < 0:
            size = len(self._mv) - self._pos
        chunk = self._mv[self._pos:self._pos + size]
        self._pos += len(chunk)
        return bytes(chunk)

    def peek(self, size: int) -> bytes:
        return bytes(self._mv[self._pos:self._pos + size])

    def mappable(self) -> bool:
        return True

    def buffer(self) -> memoryview:
        return self._mv

    def seekable(self) -> bool:
        return True

    def seek(self, pos: int, whence: int = io.SEEK_SET) -> int:
        self._pos = _clamp_seek(pos, whence, self._pos, len(self._mv))
        return self._pos

    def tell(self) -> int:
        return self._pos

class MmapSource(ByteSource):
    """Memory-mapped, read-only file source. Zero-copy: the whole file is
    addressable, so the native parser can walk it in place.

    Drop any ``buffer()``/``np.frombuffer`` views before :meth:`close`.
    """

    def __init__(self, path: str | Path) -> None:
        path = Path(path)
        self._f = open(path, "rb")
        try:
            self._mm = mmap.mmap(self._f.fileno(), 0, access=mmap.ACCESS_READ)
        except (ValueError, OSError):
            self._f.close()
            raise  # empty file / unmappable -- caller falls back to StreamSource
        self.name = path.name
        self._pos = 0

    def read(self, size: int = -1) -> bytes:
        if size < 0:
            size = len(self._mm) - self._pos
        chunk = self._mm[self._pos:self._pos + size]
        self._pos += len(chunk)
        return chunk

    def peek(self, size: int) -> bytes:
        return bytes(self._mm[self._pos:self._pos + size])

    def mappable(self) -> bool:
        return True

    def buffer(self) -> memoryview:
        return memoryview(self._mm)

    def seekable(self) -> bool:
        return True

    def seek(self, pos: int, whence: int = io.SEEK_SET) -> int:
        self._pos = _clamp_seek(pos, whence, self._pos, len(self._mm))
        return self._pos

    def tell(self) -> int:
        return self._pos

    def close(self) -> None:
        # Raises BufferError if a caller still holds a buffer()/frombuffer view.
        self._mm.close()
        self._f.close()

def make_source(inp: "Path | str | bytes | io.BufferedIOBase", *, mmap_files: bool = True) -> ByteSource:
    """Normalise ``inp`` into a :class:`ByteSource`.

    Accepts a path (str/Path), an in-memory buffer (bytes/bytearray/memoryview),
    a ``BytesIO``, any object with a binary ``read`` (e.g. ``BufferedReader``),
    or an already-constructed :class:`ByteSource` (returned as-is).

    Regular files are memory-mapped by default (zero-copy); set
    ``mmap_files=False`` or fall back automatically for empty/unmappable files.
    """
    if isinstance(inp, ByteSource):
        return inp
    if isinstance(inp, (str, Path)):
        p = Path(inp)
        if not p.is_file():
            raise FileNotFoundError(f"File {p} does not exist")
        if is_compressed_path(p):
            # A compressed file cannot be mmap'd; wrap the decompressing stream.
            # Keep the full name (incl. the compression suffix) so format
            # detection can strip it back to the inner extension.
            return StreamSource(open_compressed(p, "rb"), name=p.name, owns=True)
        if mmap_files and p.stat().st_size > 0:
            try:
                return MmapSource(p)
            except (ValueError, OSError):
                pass
        return StreamSource(open(p, "rb"), name=p.name, owns=True)
    if isinstance(inp, (bytes, bytearray, memoryview)):
        return BufferSource(inp)
    if isinstance(inp, io.BytesIO):
        return BufferSource(inp.getbuffer())  # zero-copy view into the BytesIO
    if hasattr(inp, "read"):
        return StreamSource(inp)
    raise TypeError(f"Cannot create a ByteSource from {type(inp).__name__}")
