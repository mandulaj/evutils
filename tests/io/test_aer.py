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
