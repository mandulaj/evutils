"""External-trigger decoding tests for the EVT2 / EVT2.1 / EVT3 / EVT4 parsers.

The encoders do not write trigger packets, so these tests hand-craft
EXT_TRIGGER words from the Prophesee format specs and splice them into
encoder-generated files:

* EVT2   (uint32): type=0xA in bits 31..28, ts low 6 bits in 27..22,
  id in bits 11..8, value in bit 0.
* EVT2.1 (uint64): type=0xA in bits 31..28, ts low 6 bits in 27..22,
  id in bits 44..40, value in bit 32.
* EVT3   (uint16): type=0xA in bits 15..12, id in bits 11..8, value in bit 0;
  timestamp comes from the running TIME_HIGH/TIME_LOW state.
* EVT4   (uint32): type=0x9 in bits 31..28, ts low 6 bits in 27..22,
  id in bits 12..8 (5-bit), value in bit 0; TIME_HIGH is type=0xE.
"""
import warnings

import numpy as np

from typing import Any


####################################
# word crafting helpers
####################################

def _evt2_time_high(t: int) -> int:
    return (0x8 << 28) | ((t >> 6) & 0x0FFFFFFF)


def _evt2_trigger(t: int, tid: int, value: int) -> int:
    return (0xA << 28) | ((t & 0x3F) << 22) | ((tid & 0xF) << 8) | (value & 1)


def _evt2_cd(t: int, x: int, y: int, p: int) -> int:
    return ((p & 1) << 28) | ((t & 0x3F) << 22) | ((x & 0x7FF) << 11) | (y & 0x7FF)


def _evt21_trigger_words(t: int, tid: int, value: int) -> Any:
    th = (0x8 << 28) | ((t >> 6) & 0x0FFFFFFF)
    tr = (0xA << 28) | ((t & 0x3F) << 22) | ((tid & 0x1F) << 40) | ((value & 1) << 32)
    return np.array([th, tr], dtype=np.uint64)


def _evt4_time_high(t: int) -> int:
    return (0xE << 28) | ((t >> 6) & 0x0FFFFFFF)


def _evt4_trigger(t: int, tid: int, value: int) -> int:
    # EVT4 EXT_TRIGGER=0x9: value bit 0, id bits 12..8 (5-bit), ts low bits 27..22.
    return (0x9 << 28) | ((t & 0x3F) << 22) | ((tid & 0x1F) << 8) | (value & 1)


def _evt3_trigger_words(t: int, tid: int, value: int) -> Any:
    return np.array([
        0x8000 | ((t >> 12) & 0xFFF),   # TIME_HIGH
        0x6000 | (t & 0xFFF),           # TIME_LOW
        0xA000 | ((tid & 0xF) << 8) | (value & 1),
    ], dtype=np.uint16)


def _events(n: int = 100) -> Any:
    ev = np.zeros(n, dtype=np.dtype([('t', np.int64), ('x', np.uint16), ('y', np.uint16), ('p', np.uint8)]))
    ev['t'] = np.arange(n, dtype=np.int64) * 100  # 0 .. 9900 us
    ev['x'] = np.arange(n) % 1280
    ev['y'] = np.arange(n) % 720
    ev['p'] = np.arange(n) % 2
    return ev


# Trigger fixtures appended after the events, so timestamps must not run
# backwards: all >= the last event timestamp.
TRIGGERS = [  # (t, id, value)
    (10_000, 3, 1),
    (10_064, 3, 0),
    (20_000, 7, 1),
]


def _write_events_with_triggers(tmp_path: Any, fmt: str, trigger_words: Any) -> Any:
    """Encoder-generated file (header + events) with raw words appended."""
    from evutils.io import EventWriter
    p = tmp_path / f"trig_{fmt}.raw"
    with EventWriter(p, format=fmt) as w:
        w.write(_events())
    with open(p, "ab") as f:
        f.write(trigger_words.tobytes())
    return p


####################################
# read paths
####################################

def test_evt3_trigger_decode(tmp_path: Any) -> None:
    from evutils.io import EventReader
    words = np.concatenate([_evt3_trigger_words(t, i, v) for t, i, v in TRIGGERS])
    p = _write_events_with_triggers(tmp_path, "evt3", words)

    with EventReader(p, ext_trigger=True) as r:
        out = r.read_all()
    assert isinstance(out, tuple)
    ev, tr = out
    assert np.array_equal(ev.t, _events()['t'])
    assert np.array_equal(tr.t, [t for t, _, _ in TRIGGERS])
    assert np.array_equal(tr.id, [i for _, i, _ in TRIGGERS])
    assert np.array_equal(tr.p, [v for _, _, v in TRIGGERS])


def test_evt21_trigger_decode(tmp_path: Any) -> None:
    from evutils.io import EventReader
    words = np.concatenate([_evt21_trigger_words(t, i, v) for t, i, v in TRIGGERS])
    p = _write_events_with_triggers(tmp_path, "evt21", words)

    with EventReader(p, ext_trigger=True) as r:
        ev, tr = r.read_all()
    assert np.array_equal(ev.t, _events()['t'])
    assert np.array_equal(tr.t, [t for t, _, _ in TRIGGERS])
    assert np.array_equal(tr.id, [i for _, i, _ in TRIGGERS])
    assert np.array_equal(tr.p, [v for _, _, v in TRIGGERS])


def test_evt2_trigger_timestamps(tmp_path: Any) -> None:
    from evutils.io import EventReader
    words = np.array(
        [w for t, i, v in TRIGGERS for w in (_evt2_time_high(t), _evt2_trigger(t, i, v))],
        dtype=np.uint32,
    )
    p = _write_events_with_triggers(tmp_path, "evt2", words)

    with EventReader(p, ext_trigger=True) as r:
        ev, tr = r.read_all()
    assert np.array_equal(ev.t, _events()['t'])
    assert np.array_equal(tr.t, [t for t, _, _ in TRIGGERS])


def test_evt2_trigger_id_and_value(tmp_path: Any) -> None:
    from evutils.io import EventReader
    words = np.array(
        [w for t, i, v in TRIGGERS for w in (_evt2_time_high(t), _evt2_trigger(t, i, v))],
        dtype=np.uint32,
    )
    p = _write_events_with_triggers(tmp_path, "evt2", words)

    with EventReader(p, ext_trigger=True) as r:
        _, tr = r.read_all()
    assert np.array_equal(tr.id, [i for _, i, _ in TRIGGERS])
    assert np.array_equal(tr.p, [v for _, _, v in TRIGGERS])


def test_evt4_trigger_decode(tmp_path: Any) -> None:
    from evutils.io import EventReader
    words = np.array(
        [w for t, i, v in TRIGGERS for w in (_evt4_time_high(t), _evt4_trigger(t, i, v))],
        dtype=np.uint32,
    )
    p = _write_events_with_triggers(tmp_path, "evt4", words)

    with EventReader(p, ext_trigger=True) as r:
        ev, tr = r.read_all()
    assert np.array_equal(ev.t, _events()['t'])
    assert np.array_equal(tr.t, [t for t, _, _ in TRIGGERS])
    assert np.array_equal(tr.id, [i for _, i, _ in TRIGGERS])
    assert np.array_equal(tr.p, [v for _, _, v in TRIGGERS])


def test_evt4_trigger_5bit_id(tmp_path: Any) -> None:
    """EVT4 trigger id is 5 bits (0..31), wider than EVT2's 4-bit field."""
    from evutils.io import EventReader
    trigs = [(10_000, 17, 1), (10_064, 31, 0)]  # ids > 15 need the 5th bit
    words = np.array(
        [w for t, i, v in trigs for w in (_evt4_time_high(t), _evt4_trigger(t, i, v))],
        dtype=np.uint32,
    )
    p = _write_events_with_triggers(tmp_path, "evt4", words)
    with EventReader(p, ext_trigger=True) as r:
        _, tr = r.read_all()
    assert np.array_equal(tr.id, [i for _, i, _ in trigs])
    assert np.array_equal(tr.p, [v for _, _, v in trigs])


####################################
# windowed reads: triggers land in the window covering their timestamp
####################################

def test_evt2_trigger_windowing(tmp_path: Any) -> None:
    """Hand-craft a full interleaved stream (events + triggers) and read it in
    delta_t windows: each window's triggers must fall inside its time span."""
    from evutils.io import EventReader, EventWriter

    ev = _events(100)  # t = 0 .. 9900
    trig = [(1_500, 1, 1), (4_500, 2, 0), (8_500, 3, 1)]

    # header only
    p = tmp_path / "windowed_evt2.raw"
    w = EventWriter(p, format="evt2")
    w.init()
    w.close()

    # interleave CD and trigger words in timestamp order
    items = sorted(
        [("cd", int(e['t']), e) for e in ev] + [("tr", t, (t, i, v)) for t, i, v in trig],
        key=lambda kv: kv[1],
    )
    words, last_high = [], -1
    for kind, t, payload in items:
        if (t >> 6) != last_high:
            last_high = t >> 6
            words.append(_evt2_time_high(t))
        if kind == "cd":
            e = payload
            words.append(_evt2_cd(t, int(e['x']), int(e['y']), int(e['p'])))
        else:
            words.append(_evt2_trigger(*payload))
    with open(p, "ab") as f:
        f.write(np.array(words, dtype=np.uint32).tobytes())

    seen_ev, seen_tr = 0, []
    with EventReader(p, ext_trigger=True, mode="delta_t", delta_t=2_000) as r:
        for chunk, triggers in r:
            if len(chunk) == 0 and len(triggers) == 0:
                continue
            if len(chunk) > 0 and len(triggers) > 0:
                lo, hi = int(chunk.t[0]), int(chunk.t[-1])
                # window is 2 ms wide; triggers must lie within its span
                assert all(lo - 2_000 < t <= hi + 2_000 for t in triggers.t)
            seen_ev += len(chunk)
            seen_tr.extend(zip(triggers.t.tolist(), triggers.id.tolist(), triggers.p.tolist()))

    assert seen_ev == len(ev)
    assert seen_tr == trig


####################################
# plumbing
####################################

def test_triggers_dropped_when_disabled(tmp_path: Any) -> None:
    """ext_trigger=False (default): plain EventArray, trigger words skipped."""
    from evutils.io import EventReader
    words = np.concatenate([_evt3_trigger_words(t, i, v) for t, i, v in TRIGGERS])
    p = _write_events_with_triggers(tmp_path, "evt3", words)

    with EventReader(p) as r:
        out = r.read_all()
    assert not isinstance(out, tuple)
    assert len(out) == len(_events())


def test_no_triggers_yields_empty_array(tmp_path: Any) -> None:
    from evutils.io import EventReader, EventWriter
    p = tmp_path / "no_trig.raw"
    with EventWriter(p, format="evt3") as w:
        w.write(_events())
    with EventReader(p, ext_trigger=True) as r:
        ev, tr = r.read_all()
    assert len(ev) == len(_events())
    assert len(tr) == 0


def test_ext_trigger_unsupported_decoder_warns(tmp_path: Any) -> None:
    from evutils.io import EventReader, EventWriter
    p = tmp_path / "events.csv"
    with EventWriter(p) as w:
        w.write(_events())
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        r = EventReader(p, ext_trigger=True)
        r.close()
    assert any("does not support" in str(w.message) for w in caught)
