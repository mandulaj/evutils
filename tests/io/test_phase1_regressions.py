"""Regression tests for the Phase-1 correctness fixes.

Each test pins one previously-broken behaviour:
- tiny chunk_size used to underflow the C parsers' capacity headroom
  (size_t wrap => heap corruption hazard);
- delta_t fast-path overshoot (``_dt_carry``) used to be invisible to
  mixed-mode reads and ``read_all()`` (silent event loss);
- ``max_events`` was not actually enforced within one C delta_t call;
- ``EventWriter.write()`` silently dropped its ``triggers`` argument;
- EVT2.1 with a non-legacy endianness header decoded silently into garbage.
"""
import numpy as np
import pytest

from typing import Any


def _uniform_events(n: int = 5000, dt: int = 13) -> Any:
    ev = np.zeros(n, dtype=np.dtype([('t', np.int64), ('x', np.uint16),
                                     ('y', np.uint16), ('p', np.uint8)]))
    ev['t'] = np.arange(n, dtype=np.int64) * dt
    ev['x'] = np.arange(n) % 1280
    ev['y'] = np.arange(n) % 720
    ev['p'] = np.arange(n) % 2
    return ev


def _write(tmp_path: Any, fmt: str, ev: Any) -> Any:
    from evutils.io import EventWriter
    p = tmp_path / f"reg_{fmt}.raw"
    with EventWriter(p, format=fmt) as w:
        w.write(ev)
    return p


####################################
# C1: tiny chunk_size
####################################

def test_tiny_chunk_size_rejected(tmp_path: Any) -> None:
    from evutils.io import EventReader
    p = _write(tmp_path, "evt3", _uniform_events(100))
    with pytest.raises(ValueError, match="chunk_size"):
        EventReader(p, n_events=10, chunk_size=16).read()


def test_minimal_chunk_size_decodes(tmp_path: Any) -> None:
    """chunk_size=128 (the minimum) must decode the whole stream, not hang or
    corrupt: the parser's capacity headroom (64) stays below the buffer size."""
    from evutils.io import EventReader
    ev = _uniform_events(1000)
    p = _write(tmp_path, "evt3", ev)
    total = 0
    with EventReader(p, n_events=200, chunk_size=128) as r:
        for c in r:
            total += len(c)
    assert total == len(ev)


####################################
# C3: carry visibility on mixed paths
####################################

@pytest.mark.parametrize("fmt", ["evt2", "evt3"])
def test_mixed_delta_t_then_read_all_loses_nothing(tmp_path: Any, fmt: str) -> None:
    from evutils.io import EventReader
    ev = _uniform_events(5000)
    p = _write(tmp_path, fmt, ev)
    got = []
    with EventReader(p, delta_t=10_000) as r:
        # a few fast-path windows...
        for _ in range(3):
            got.append(np.asarray(r.read()).copy())
        # ...then drain the rest at once: the fast-path overshoot carry must be
        # folded in, not dropped.
        got.append(np.asarray(r.read_all()).copy())
    out = np.concatenate([g for g in got if len(g)])
    assert len(out) == len(ev)
    assert np.array_equal(out['t'], ev['t'])


@pytest.mark.parametrize("fmt", ["evt2", "evt3"])
def test_mixed_delta_t_then_n_events_override(tmp_path: Any, fmt: str) -> None:
    from evutils.io import EventReader
    ev = _uniform_events(5000)
    p = _write(tmp_path, fmt, ev)
    got = []
    with EventReader(p, delta_t=10_000) as r:
        got.append(np.asarray(r.read()).copy())          # fast path window
        got.append(np.asarray(r.read(n_events=700)).copy())  # override -> acc path
        while True:                                       # drain
            c = np.asarray(r.read())
            if len(c) == 0:
                break
            got.append(c.copy())
    out = np.concatenate([g for g in got if len(g)])
    assert len(out) == len(ev)
    assert np.array_equal(out['t'], ev['t'])


####################################
# C4: max_events honoured in delta_t mode
####################################

def test_delta_t_windows_respect_max_events(tmp_path: Any) -> None:
    from evutils.io import EventReader
    ev = _uniform_events(5000)
    p = _write(tmp_path, "evt3", ev)
    total = 0
    # One delta_t window would span the whole file; the cap must split it.
    with EventReader(p, delta_t=10_000_000, max_events=1000) as r:
        for c in r:
            assert len(c) <= 1000
            total += len(c)
    assert total == len(ev)


####################################
# C5: EventWriter.write forwards triggers
####################################

def test_writer_warns_on_unsupported_triggers(tmp_path: Any) -> None:
    """No encoder implements trigger *encoding* yet, so the honest contract is:
    EventWriter forwards triggers to the encoder and warns LOUDLY (once) when
    the encoder cannot store them -- never a silent drop. When an encoder gains
    trigger support (SUPPORTS_WRITE_TRIGGERS = True) this becomes a real
    round-trip test."""
    from evutils.io import EventReader, EventWriter
    from evutils.types import TriggerArray
    ev = _uniform_events(2000)
    tr = TriggerArray(t=[100, 5000, 20_000], p=[1, 0, 1], id=[0, 1, 0])
    p = tmp_path / "trig.raw"
    with EventWriter(p, format="evt3") as w:
        with pytest.warns(UserWarning, match="triggers"):
            w.write(ev, triggers=tr)
        w.write(ev, triggers=tr)  # warning fires only once
    with EventReader(p) as r:
        out = r.read_all()
    assert len(out) == 2 * len(ev)  # events themselves are unaffected


####################################
# C9: EVT2.1 endianness guard
####################################

def test_evt21_non_legacy_endianness_rejected() -> None:
    from evutils.io import EventReader
    hdr = b"% evt 2.1\n% endianness little\n% geometry 1280x720\n% end\n"
    with pytest.raises(NotImplementedError, match="endianness"):
        EventReader(hdr).read()


def test_evt21_legacy_endianness_accepted(tmp_path: Any) -> None:
    from evutils.io import EventReader
    ev = _uniform_events(500)
    p = _write(tmp_path, "evt21", ev)  # writer emits "% endianness legacy"
    with EventReader(p) as r:
        out = r.read_all()
    assert len(out) == len(ev)
