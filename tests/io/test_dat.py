from typing import Any
import numpy as np
import pytest

def test_DAT_writer(tmp_path: Any, test_events: Any) -> None:
    from evutils.io import EventReader, EventWriter

    p = tmp_path / "events.dat"
    with EventWriter(p) as writer:
        writer.write(test_events)

    assert p.is_file()
    assert p.stat().st_size > 0

    with EventReader(p) as reader:
        events = reader.read()
    assert np.array_equal(events, test_events)


def test_DAT_matches_expelliarmus(tmp_path: Any, test_events: Any) -> None:
    """Our DAT output must be readable byte-for-byte by the reference reader,
    and our EventReader must decode it exactly the same as the reference reader."""
    pytest.importorskip("expelliarmus")
    from expelliarmus import Wizard # type: ignore

    from evutils.io import EventWriter, EventReader

    p = tmp_path / "events.dat"
    with EventWriter(p) as writer:
        writer.write(test_events)

    # 1. Read with expelliarmus
    exp_arr = Wizard(encoding="dat").read(str(p))
    
    # 2. Read with evutils
    with EventReader(p) as reader:
        evutils_arr = reader.read()
    assert not isinstance(evutils_arr, tuple)

    # Compare them directly
    assert len(evutils_arr) == len(exp_arr)
    assert np.array_equal(evutils_arr["t"], exp_arr["t"])
    assert np.array_equal(evutils_arr["x"], exp_arr["x"])
    assert np.array_equal(evutils_arr["y"], exp_arr["y"])
    assert np.array_equal(evutils_arr["p"], exp_arr["p"])


def test_DAT_timestamp_overflow(tmp_path: Any) -> None:
    """DAT timestamps are 32-bit us on disk (~71 min)
    the decoder must extend
    them past the wrap."""
    from evutils.io import EventReader, EventWriter
    from evutils.types import Event_dtype

    t = np.array([2**32 - 30, 2**32 - 10, 2**32 + 10, 2**32 + 30, 2**33 + 5], dtype=np.int64)
    ev = np.zeros(len(t), dtype=Event_dtype)
    ev['t'] = t
    ev['x'] = np.arange(len(t))
    ev['y'] = np.arange(len(t))
    ev['p'] = [0, 1, 0, 1, 1]

    p = tmp_path / "overflow.dat"
    with EventWriter(p) as w:
        w.write(ev)
    with EventReader(p) as r:
        out = r.read_all()
    assert not isinstance(out, tuple)
    assert np.array_equal(out["t"], t)


def test_DAT_coordinate_extremes(tmp_path: Any) -> None:
    """DAT coordinates are 14-bit
    extremes must survive."""
    from evutils.io import EventReader, EventWriter
    from evutils.types import Event_dtype

    ev = np.zeros(4, dtype=Event_dtype)
    ev['t'] = [0, 1, 2, 3]
    ev['x'] = [0, 2**14 - 1, 5, 2**14 - 2]
    ev['y'] = [2**14 - 1, 0, 6, 2**14 - 3]
    ev['p'] = [1, 0, 1, 0]

    p = tmp_path / "coords.dat"
    with EventWriter(p) as w:
        w.write(ev)
    with EventReader(p) as r:
        out = r.read_all()
    assert not isinstance(out, tuple)
    assert np.array_equal(np.asarray(out), ev)

def test_DAT_empty_file(tmp_path: Any) -> None:
    from evutils.io import EventReader
    p = tmp_path / "empty.dat"
    p.touch()
    with EventReader(p) as r:
        assert len(r.read()) == 0

def test_DAT_header_only(tmp_path: Any) -> None:
    from evutils.io import EventReader
    p = tmp_path / "header_only.dat"
    p.write_bytes(b"% Header\n")
    with EventReader(p) as r:
        assert len(r.read()) == 0

def test_DAT_unsupported_event_size(tmp_path: Any) -> None:
    from evutils.io import EventReader
    p = tmp_path / "unsupported_size.dat"
    # 0x0C = CD, size 0x10 (unsupported)
    p.write_bytes(b"% Header\n\x0c\x10")
    with pytest.raises(NotImplementedError, match="DAT event size 16 not supported"):
        with EventReader(p) as r:
            r.read()

def test_DAT_truncated_payload(tmp_path: Any) -> None:
    # 8 bytes per event. Give 4 bytes.
    from evutils.io import EventReader
    p = tmp_path / "truncated.dat"
    p.write_bytes(b"% Header\n\x0c\x08\x01\x02\x03\x04")
    with EventReader(p) as r:
        assert len(r.read()) == 0

def test_DAT_truncated_header_missing_bytes(tmp_path: Any) -> None:
    from evutils.io import EventReader
    p = tmp_path / "truncated_header.dat"
    p.write_bytes(b"% Header\n\x0c") # 1 byte left after header
    with EventReader(p) as r:
        assert len(r.read()) == 0
