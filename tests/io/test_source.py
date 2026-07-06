"""Unit tests for the ByteSource hierarchy (io/_source.py).

Covers BufferSource, StreamSource, MmapSource and the make_source() factory:
sequential reads, non-consuming peek, seek/tell clamping, zero-copy mapping,
lifetime rules (views must be dropped before close), and factory dispatch.
"""
import io

import numpy as np
import pytest

from typing import Any

from evutils.io._source import (
    BufferSource,
    MmapSource,
    StreamSource,
    make_source,
)

PAYLOAD = b"0123456789abcdef"


####################################
# BufferSource
####################################

def test_buffersource_sequential_read() -> None:
    src = BufferSource(PAYLOAD)
    assert src.read(4) == b"0123"
    assert src.read(4) == b"4567"
    assert src.read(-1) == b"89abcdef"
    assert src.read(4) == b""  # EOF


def test_buffersource_peek_does_not_consume() -> None:
    src = BufferSource(PAYLOAD)
    assert src.peek(4) == b"0123"
    assert src.peek(4) == b"0123"
    assert src.read(4) == b"0123"
    assert src.peek(4) == b"4567"


def test_buffersource_peek_past_end() -> None:
    src = BufferSource(b"ab")
    assert src.peek(10) == b"ab"


def test_buffersource_seek_tell() -> None:
    src = BufferSource(PAYLOAD)
    assert src.seekable()
    assert src.tell() == 0
    assert src.seek(8) == 8
    assert src.tell() == 8
    assert src.read(2) == b"89"
    assert src.seek(-2, io.SEEK_CUR) == 8
    assert src.seek(-4, io.SEEK_END) == len(PAYLOAD) - 4
    assert src.read(-1) == b"cdef"


def test_buffersource_seek_clamps() -> None:
    src = BufferSource(PAYLOAD)
    assert src.seek(-100) == 0
    assert src.seek(1000) == len(PAYLOAD)
    assert src.read(1) == b""
    with pytest.raises(ValueError):
        src.seek(0, whence=42)


def test_buffersource_mappable_zero_copy() -> None:
    data = bytearray(PAYLOAD)
    src = BufferSource(data)
    assert src.mappable()
    mv = src.buffer()
    assert bytes(mv) == PAYLOAD
    data[0] = ord(b"X")  # same storage: mutation is visible through the view
    assert bytes(mv[:1]) == b"X"


def test_buffersource_accepts_memoryview_and_bytesio_buffer() -> None:
    assert BufferSource(memoryview(PAYLOAD)).read(-1) == PAYLOAD
    bio = io.BytesIO(PAYLOAD)
    assert BufferSource(bio.getbuffer()).read(-1) == PAYLOAD


def test_buffersource_readline_generic() -> None:
    src = BufferSource(b"line1\nline2\nno-newline")
    assert src.readline() == b"line1\n"
    assert src.readline() == b"line2\n"
    assert src.readline() == b"no-newline"
    assert src.readline() == b""


def test_buffersource_reset() -> None:
    src = BufferSource(PAYLOAD)
    src.read(8)
    src.reset()
    assert src.tell() == 0
    assert src.read(4) == b"0123"


####################################
# StreamSource
####################################

class _ReadOnlyStream:
    """Minimal non-seekable, non-peekable binary stream (pipe-like)."""

    def __init__(self, data: bytes) -> None:
        self._bio = io.BytesIO(data)

    def read(self, size: int = -1) -> bytes:
        return self._bio.read(size)

    def seekable(self) -> bool:
        return False


def test_streamsource_requires_read() -> None:
    with pytest.raises(TypeError):
        StreamSource(object())


def test_streamsource_read_and_seek_via_bytesio() -> None:
    src = StreamSource(io.BytesIO(PAYLOAD))
    assert src.read(4) == b"0123"
    assert src.seekable()
    assert src.tell() == 4
    src.seek(0)
    assert src.read(-1) == PAYLOAD
    assert not src.mappable()
    with pytest.raises(io.UnsupportedOperation):
        src.buffer()


def test_streamsource_peek_fallback_via_seek() -> None:
    # BytesIO has no peek() but is seekable: peek must restore the position.
    src = StreamSource(io.BytesIO(PAYLOAD))
    src.read(2)
    assert src.peek(4) == b"2345"
    assert src.tell() == 2
    assert src.read(4) == b"2345"


def test_streamsource_peek_native(tmp_path: Any) -> None:
    # BufferedReader exposes peek() directly.
    p = tmp_path / "data.bin"
    p.write_bytes(PAYLOAD)
    with open(p, "rb") as f:
        src = StreamSource(f)
        assert src.peek(4) == b"0123"
        assert src.read(4) == b"0123"
        assert src.name == str(p)


def test_streamsource_nonseekable_peek_raises() -> None:
    src = StreamSource(_ReadOnlyStream(PAYLOAD))
    assert not src.seekable()
    with pytest.raises(io.UnsupportedOperation):
        src.peek(4)
    # sequential reading still works
    assert src.read(4) == b"0123"


def test_streamsource_readline_delegates() -> None:
    src = StreamSource(io.BytesIO(b"a\nb\n"))
    assert src.readline() == b"a\n"
    assert src.readline() == b"b\n"


def test_streamsource_close_ownership() -> None:
    owned = io.BytesIO(PAYLOAD)
    StreamSource(owned, owns=True).close()
    assert owned.closed

    borrowed = io.BytesIO(PAYLOAD)
    StreamSource(borrowed, owns=False).close()
    assert not borrowed.closed


####################################
# MmapSource
####################################

@pytest.fixture
def payload_file(tmp_path: Any) -> Any:
    p = tmp_path / "payload.bin"
    p.write_bytes(PAYLOAD)
    return p


def test_mmapsource_read_peek_seek(payload_file: Any) -> None:
    with MmapSource(payload_file) as src:
        assert src.mappable() and src.seekable()
        assert src.name == payload_file.name
        assert src.peek(4) == b"0123"
        assert src.read(4) == b"0123"
        assert src.tell() == 4
        src.seek(-4, io.SEEK_END)
        assert src.read(-1) == b"cdef"


def test_mmapsource_buffer_zero_copy(payload_file: Any) -> None:
    src = MmapSource(payload_file)
    mv = src.buffer()
    arr = np.frombuffer(mv, dtype=np.uint8)
    assert bytes(arr.tobytes()) == PAYLOAD
    # Views alias the mapping: they must be dropped before close().
    del arr
    mv.release()
    src.close()


def test_mmapsource_close_with_live_view_raises(payload_file: Any) -> None:
    src = MmapSource(payload_file)
    mv = src.buffer()
    with pytest.raises(BufferError):
        src.close()
    mv.release()
    src.close()


def test_mmapsource_empty_file_raises(tmp_path: Any) -> None:
    p = tmp_path / "empty.bin"
    p.touch()
    with pytest.raises((ValueError, OSError)):
        MmapSource(p)


####################################
# make_source factory
####################################

def test_make_source_path_maps(payload_file: Any) -> None:
    src = make_source(payload_file)
    assert isinstance(src, MmapSource)
    assert src.read(-1) == PAYLOAD
    src.close()
    # str path works too
    src = make_source(str(payload_file))
    assert isinstance(src, MmapSource)
    src.close()


def test_make_source_path_no_mmap(payload_file: Any) -> None:
    src = make_source(payload_file, mmap_files=False)
    assert isinstance(src, StreamSource)
    assert src.read(-1) == PAYLOAD
    src.close()


def test_make_source_empty_file_falls_back_to_stream(tmp_path: Any) -> None:
    p = tmp_path / "empty.bin"
    p.touch()
    src = make_source(p)
    assert isinstance(src, StreamSource)
    assert src.read(-1) == b""
    src.close()


def test_make_source_missing_file() -> None:
    with pytest.raises(FileNotFoundError):
        make_source("/nonexistent/definitely_missing.raw")


def test_make_source_buffers() -> None:
    assert isinstance(make_source(PAYLOAD), BufferSource)
    assert isinstance(make_source(bytearray(PAYLOAD)), BufferSource)
    assert isinstance(make_source(memoryview(PAYLOAD)), BufferSource)


def test_make_source_bytesio_zero_copy() -> None:
    src = make_source(io.BytesIO(PAYLOAD))
    assert isinstance(src, BufferSource)
    assert src.read(-1) == PAYLOAD


def test_make_source_readable_object() -> None:
    src = make_source(_ReadOnlyStream(PAYLOAD))
    assert isinstance(src, StreamSource)
    assert src.read(-1) == PAYLOAD


def test_make_source_passthrough() -> None:
    inner = BufferSource(PAYLOAD)
    assert make_source(inner) is inner


def test_make_source_rejects_unknown_type() -> None:
    with pytest.raises(TypeError):
        make_source(12345)
