"""Tests for lightweight EventArray metadata (sensor_size) and its io wiring."""
import numpy as np
import pytest

from evutils.io import EventReader, EventWriter
from evutils.types import Event_dtype, EventArray


def _events(n=3000, w=320, h=240):
    return EventArray(
        t=np.arange(n, dtype=np.int64) * 10,
        x=(np.arange(n) % w).astype(np.uint16),
        y=(np.arange(n) % h).astype(np.uint16),
        p=(np.arange(n) % 2).astype(np.uint8),
    )


# --------------------------------------------------------------------------- #
# Core type behaviour
# --------------------------------------------------------------------------- #

def test_metadata_defaults_none_and_roundtrips():
    e = EventArray(t=[1], x=[2], y=[3], p=[1])
    assert e.metadata is None
    assert e.sensor_size is None

    e.sensor_size = (64, 48)
    assert e.metadata == {"sensor_size": (64, 48)}
    assert e.sensor_size == (64, 48)


def test_metadata_survives_slice_and_subset():
    e = EventArray(t=[1, 2, 3], x=[1, 2, 3], y=[1, 2, 3], p=[1, 0, 1],
                   metadata={"sensor_size": (64, 48)})
    assert e[:2].sensor_size == (64, 48)
    assert e[["t", "x"]].sensor_size == (64, 48)


def test_copy_isolates_metadata():
    e = EventArray(t=[1], x=[2], y=[3], p=[1], metadata={"sensor_size": (64, 48)})
    c = e.copy()
    c.metadata["sensor_size"] = (1, 1)
    assert e.sensor_size == (64, 48)   # original untouched
    assert c.sensor_size == (1, 1)


def test_from_aos_accepts_metadata():
    aos = np.zeros(3, dtype=Event_dtype)
    e = EventArray.from_aos(aos, metadata={"sensor_size": (10, 20)})
    assert e.sensor_size == (10, 20)


# --------------------------------------------------------------------------- #
# Reader populates / Writer consumes
# --------------------------------------------------------------------------- #

def test_reader_populates_sensor_size(tmp_path):
    p = tmp_path / "geom.raw"
    with EventWriter(p, format="evt3", width=640, height=480) as w:
        w.write(_events())

    assert EventReader(p).read_all().sensor_size == (640, 480)

    r = EventReader(p, n_events=500)
    assert r.read().sensor_size == (640, 480)

    for chunk in EventReader(p, n_events=500):
        assert chunk.sensor_size == (640, 480)
        break


def test_writer_infers_dims_from_metadata(tmp_path):
    ev = _events(w=320, h=240)
    ev.sensor_size = (320, 240)
    p = tmp_path / "infer.raw"
    with EventWriter(p, format="evt3") as w:   # no width/height given
        w.write(ev)
    assert EventReader(p).shape() == (320, 240)


def test_writer_explicit_dims_override_metadata(tmp_path):
    ev = _events()
    ev.sensor_size = (320, 240)
    p = tmp_path / "explicit.raw"
    with EventWriter(p, format="evt3", width=100, height=50) as w:
        w.write(ev)
    assert EventReader(p).shape() == (100, 50)


def test_writer_falls_back_without_metadata(tmp_path):
    p = tmp_path / "fallback.raw"
    with EventWriter(p, format="evt3") as w:
        w.write(_events())          # no sensor_size metadata
    assert EventReader(p).shape() == (1280, 720)


def test_read_write_read_roundtrip_preserves_geometry(tmp_path):
    """Geometry read off one file flows into the next via metadata alone."""
    src = tmp_path / "src.raw"
    with EventWriter(src, format="evt3", width=346, height=260) as w:
        w.write(_events(w=346, h=260))

    events = EventReader(src).read_all()
    assert events.sensor_size == (346, 260)

    dst = tmp_path / "dst.raw"
    with EventWriter(dst, format="evt3") as w:   # dims inferred from events
        w.write(events)
    assert EventReader(dst).shape() == (346, 260)
