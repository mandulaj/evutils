import numpy as np

from evutils.types import Event_dtype


def test_AER_writer(tmp_path):
    """AER carries no timestamps and only 9-bit coords, so round-trip preserves
    x/y/p (with t == 0) for events that fit in 512x512."""
    from evutils.io import EventReader, EventWriter

    np.random.seed(7)
    N = 1000
    ev = np.zeros(N, dtype=Event_dtype)
    ev["x"] = np.random.randint(0, 320, N)
    ev["y"] = np.random.randint(0, 320, N)
    ev["p"] = np.random.randint(0, 2, N)

    p = tmp_path / "events.aer"
    with EventWriter(p) as writer:
        writer.write(ev)

    assert p.is_file()
    assert p.stat().st_size == N * 4  # 4 bytes/event, no header

    with EventReader(p) as reader:
        out = reader.read()

    assert len(out) == N
    assert np.array_equal(out.x, ev["x"])
    assert np.array_equal(out.y, ev["y"])
    assert np.array_equal(out.p, ev["p"])
    assert bool(np.all(out.t == 0))  # AER has no timestamps


def _write_aer(tmp_path, n=1000, seed=7):
    from evutils.io import EventWriter

    np.random.seed(seed)
    ev = np.zeros(n, dtype=Event_dtype)
    ev["x"] = np.random.randint(0, 512, n)
    ev["y"] = np.random.randint(0, 512, n)
    ev["p"] = np.random.randint(0, 2, n)
    p = tmp_path / "events.aer"
    with EventWriter(p) as writer:
        writer.write(ev)
    return p, ev


def test_AER_timestamps_sequential(tmp_path):
    """timestamps='sequential' generates t = t_start + i * t_step, carried
    across chunks and restarted by reset()."""
    from evutils.io import EventReader

    p, _ = _write_aer(tmp_path)
    with EventReader(p, timestamps="sequential", t_start=100, t_step=5) as r:
        out = r.read_all()
    assert np.array_equal(out.t, 100 + 5 * np.arange(1000))

    with EventReader(p, n_events=300, timestamps="sequential") as r:
        ts = np.concatenate([np.asarray(c)["t"] for c in r])
        assert np.array_equal(ts, np.arange(1000))
        r.reset()
        out = r.read_all()
    assert np.array_equal(out.t, np.arange(1000))  # restarted at t_start


def test_AER_timestamps_custom(tmp_path):
    """A user-provided array is assigned positionally, in bulk and chunked."""
    from evutils.io import EventReader

    p, _ = _write_aer(tmp_path)
    custom = np.sort(np.random.randint(0, 10**9, 1000))

    with EventReader(p, timestamps=custom) as r:
        assert np.array_equal(r.read_all().t, custom)

    with EventReader(p, n_events=300, timestamps=custom) as r:
        ts = np.concatenate([np.asarray(c)["t"] for c in r])
    assert np.array_equal(ts, custom)


def test_AER_timestamps_validation(tmp_path):
    """Bad mode strings and too-short custom arrays raise ValueError."""
    import pytest

    from evutils.io import EventReader

    p, _ = _write_aer(tmp_path)
    with pytest.raises(ValueError, match="timestamps must be"):
        EventReader(p, timestamps="bogus").read_all()
    with pytest.raises(ValueError, match="custom timestamps array has"):
        EventReader(p, timestamps=np.arange(10)).read_all()
