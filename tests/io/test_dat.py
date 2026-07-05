import numpy as np
import pytest


def test_DAT_writer(tmp_path, test_events):
    from evutils.io import EventReader, EventWriter

    p = tmp_path / "events.dat"
    with EventWriter(p) as writer:
        writer.write(test_events)

    assert p.is_file()
    assert p.stat().st_size > 0

    with EventReader(p) as reader:
        events = reader.read()
    assert np.array_equal(events, test_events)


def test_DAT_matches_expelliarmus(tmp_path, test_events):
    """Our DAT output must be readable byte-for-byte by the reference reader."""
    pytest.importorskip("expelliarmus")
    from expelliarmus import Wizard

    from evutils.io import EventWriter

    p = tmp_path / "events.dat"
    with EventWriter(p) as writer:
        writer.write(test_events)

    arr = Wizard(encoding="dat").read(str(p))
    assert len(arr) == len(test_events)
    assert np.array_equal(arr["t"], test_events["t"])
    assert np.array_equal(arr["x"], test_events["x"])
    assert np.array_equal(arr["y"], test_events["y"])
    assert np.array_equal(arr["p"], test_events["p"])
