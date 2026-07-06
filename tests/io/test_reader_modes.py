"""Windowing-mode tests for EventReader: ``mixed`` and ``all``.

``delta_t`` / ``n_events`` modes are exercised throughout the per-format
tests; the combined (``mixed``) and whole-file (``all``) modes had no
dedicated coverage. Window semantics under test:

* n_events cutoff: exactly ``n_events`` events per chunk while available.
* delta_t cutoff: half-open window ``[start, start + delta_t)``.
* mixed: whichever cutoff is hit first wins, per chunk.
* all: one read returns everything; the next read is empty.
"""
import numpy as np
import pytest

from typing import Any


def _uniform_events(n: int = 1000, dt: int = 1000) -> Any:
    """n events at exactly one event per ``dt`` microseconds."""
    ev = np.zeros(n, dtype=np.dtype([('t', np.int64), ('x', np.uint16), ('y', np.uint16), ('p', np.uint8)]))
    ev['t'] = np.arange(n, dtype=np.int64) * dt
    ev['x'] = np.arange(n) % 1280
    ev['y'] = np.arange(n) % 720
    ev['p'] = np.arange(n) % 2
    return ev


@pytest.fixture
def uniform_file(tmp_path: Any) -> Any:
    from evutils.io import EventWriter
    ev = _uniform_events()
    p = tmp_path / "uniform.raw"
    with EventWriter(p, format="evt2") as w:
        w.write(ev)
    return p, ev


####################################
# mixed mode
####################################

def test_mixed_mode_n_events_cuts_first(uniform_file: Any) -> None:
    """delta_t generous (100 events worth), n_events tight (20): every chunk
    must be cut by the event count."""
    from evutils.io import EventReader
    p, ev = uniform_file
    with EventReader(p, mode="mixed", delta_t=100_000, n_events=20) as r:
        chunks = list(r)
    assert all(len(c) == 20 for c in chunks)
    assert sum(len(c) for c in chunks) == len(ev)
    out = np.concatenate([np.asarray(c) for c in chunks])
    assert np.array_equal(out['t'], ev['t'])


def test_mixed_mode_delta_t_cuts_first(uniform_file: Any) -> None:
    """n_events generous (500), delta_t tight (5 events worth): every chunk
    must span < delta_t and hold 5 events."""
    from evutils.io import EventReader
    p, ev = uniform_file
    with EventReader(p, mode="mixed", delta_t=5_000, n_events=500) as r:
        chunks = [c for c in r if len(c) > 0]
    assert all(len(c) == 5 for c in chunks)
    for c in chunks:
        assert c['t'][-1] - c['t'][0] < 5_000  # half-open window
    out = np.concatenate([np.asarray(c) for c in chunks])
    assert np.array_equal(out['t'], ev['t'])


def test_mixed_is_default_mode(uniform_file: Any) -> None:
    """auto mode with both parameters set resolves to mixed."""
    from evutils.io import EventReader
    p, _ = uniform_file
    with EventReader(p, delta_t=100_000, n_events=20) as r:
        first = r.read()
    assert len(first) == 20


def test_mixed_mode_alternating_cutoffs(tmp_path: Any) -> None:
    """A burst then a gap: the burst chunk is cut by n_events, the sparse
    region by delta_t."""
    from evutils.io import EventReader, EventWriter
    # 50 events in the first 50 us (burst), then 50 events at 1 ms apart.
    t = np.concatenate([np.arange(50), 1_000_000 + np.arange(50) * 1000])
    ev = _uniform_events(100)
    ev['t'] = t
    p = tmp_path / "burst.raw"
    with EventWriter(p, format="evt2") as w:
        w.write(ev)

    with EventReader(p, mode="mixed", delta_t=10_000, n_events=30) as r:
        first = r.read()
        # burst: 30 events arrive well inside 10 ms -> n_events cut
        assert len(first) == 30
        total = len(first)
        for c in r:
            # sparse tail: 10 ms holds at most 10 events -> delta_t cut
            assert len(c) <= 30
            total += len(c)
    assert total == 100


####################################
# all mode
####################################

def test_all_mode_single_read(uniform_file: Any) -> None:
    from evutils.io import EventReader
    p, ev = uniform_file
    with EventReader(p, mode="all") as r:
        out = r.read()
        assert np.array_equal(np.asarray(out), ev)
        # stream exhausted: subsequent reads are empty
        assert len(r.read()) == 0


def test_all_mode_iteration_yields_once(uniform_file: Any) -> None:
    from evutils.io import EventReader
    p, ev = uniform_file
    with EventReader(p, mode="all") as r:
        chunks = [c for c in r if len(c) > 0]
    assert len(chunks) == 1
    assert np.array_equal(np.asarray(chunks[0]), ev)


def test_all_mode_matches_read_all(uniform_file: Any) -> None:
    from evutils.io import EventReader
    p, _ = uniform_file
    with EventReader(p, mode="all") as r1, EventReader(p) as r2:
        assert np.array_equal(np.asarray(r1.read()), np.asarray(r2.read_all()))


####################################
# parameter validation
####################################

def test_explicit_mode_requires_parameter(uniform_file: Any) -> None:
    from evutils.io import EventReader
    p, _ = uniform_file
    with pytest.raises(ValueError):
        EventReader(p, mode="delta_t")  # delta_t missing
    with pytest.raises(ValueError):
        EventReader(p, mode="n_events")  # n_events missing
    with pytest.raises(ValueError):
        EventReader(p, mode="bogus")


def test_nonpositive_windows_rejected(uniform_file: Any) -> None:
    from evutils.io import EventReader
    p, _ = uniform_file
    with pytest.raises(ValueError):
        EventReader(p, n_events=0)
    with pytest.raises(ValueError):
        EventReader(p, delta_t=-1)
    with pytest.raises(TypeError):
        EventReader(p, delta_t=1.5)  # type: ignore[arg-type]
