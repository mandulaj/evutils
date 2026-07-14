"""EVT4 decoder tests.

EVT4 is a 32-bit-word format modelled on OpenEB's ``evt4`` HAL decoder. Its CD,
TIME_HIGH and EXT_TRIGGER words reuse EVT2's bit layout with distinct type codes,
so the scalar path is validated against the proven EVT2 decoder. The format's
novel mechanic is vectorised CD (CD_VEC_OFF=0xC / CD_VEC_ON=0xD): a base word
carrying x/y/ts/polarity followed by a 32-bit validity mask, emitting one event
per set bit at ``x = base_x + bit_index`` (same y/ts/polarity). No encoder emits
CD_VEC, so it is exercised here with a hand-crafted word stream.
"""
import numpy as np

from typing import Any


# ---- EVT4 word crafting (see csrc/evt4.c for the bit layout) --------------

def _time_high(t: int) -> int:
    return (0xE << 28) | ((t >> 6) & 0x0FFFFFFF)


def _cd(t: int, x: int, y: int, p: int) -> int:
    # CD_OFF=0xA / CD_ON=0xB
    return ((0xA | (p & 1)) << 28) | ((t & 0x3F) << 22) | ((x & 0x7FF) << 11) | (y & 0x7FF)


def _cd_vec_base(t: int, base_x: int, y: int, p: int) -> int:
    # CD_VEC_OFF=0xC / CD_VEC_ON=0xD; followed by a 32-bit mask word.
    return ((0xC | (p & 1)) << 28) | ((t & 0x3F) << 22) | ((base_x & 0x7FF) << 11) | (y & 0x7FF)


def _header_only(tmp_path: Any) -> Any:
    """An EVT4 file with a valid header and no payload, ready to append raw words."""
    from evutils.io import EventWriter
    p = tmp_path / "evt4_raw.raw"
    w = EventWriter(p, format="evt4")
    w.init()
    w.close()
    return p


def _append(p: Any, words: Any) -> None:
    with open(p, "ab") as f:
        f.write(np.asarray(words, dtype=np.uint32).tobytes())


# ---- scalar path: EVT4 must decode identically to EVT2 --------------------

def test_evt4_matches_evt2(tmp_path: Any, test_events: Any) -> None:
    """Same events encoded as EVT2 and EVT4 must decode to identical arrays
    (both formats share the CD / TIME_HIGH layout)."""
    from evutils.io import EventWriter, EventReader
    out = {}
    for fmt in ("evt2", "evt4"):
        p = tmp_path / f"{fmt}.raw"
        with EventWriter(p, format=fmt) as w:
            w.write(test_events)
        with EventReader(p) as r:
            out[fmt] = np.asarray(r.read_all())
    assert np.array_equal(out["evt4"], test_events), "evt4 scalar round-trip differs from input"
    assert np.array_equal(out["evt4"], out["evt2"]), "evt4 decode differs from evt2"


# ---- vectorised CD: the novel mechanic, hand-crafted ----------------------

def test_evt4_vector_cd(tmp_path: Any) -> None:
    """A CD_VEC base + 32-bit mask emits one event per set bit at base_x+bit."""
    from evutils.io import EventReader
    p = _header_only(tmp_path)

    t, base_x, y, pol = 5, 100, 50, 1
    mask = (1 << 0) | (1 << 3) | (1 << 31)  # bits 0, 3, 31 set
    _append(p, [_time_high(t), _cd_vec_base(t, base_x, y, pol), mask])

    with EventReader(p) as r:
        out = np.asarray(r.read_all())

    assert len(out) == 3, f"expected 3 events from the 3-bit mask, got {len(out)}"
    assert np.array_equal(out["x"], [100, 103, 131]), "vector x offsets wrong"
    assert np.array_equal(out["y"], [50, 50, 50]), "vector y not held constant"
    assert np.array_equal(out["p"], [1, 1, 1]), "vector polarity wrong"
    assert np.array_equal(out["t"], [5, 5, 5]), "vector timestamp wrong"


def test_evt4_vector_empty_mask(tmp_path: Any) -> None:
    """A CD_VEC with an all-zero mask emits nothing but still consumes the mask
    word, so a following scalar CD decodes correctly."""
    from evutils.io import EventReader
    p = _header_only(tmp_path)
    _append(p, [
        _time_high(7),
        _cd_vec_base(7, 200, 60, 0), 0x00000000,   # empty mask -> no events
        _cd(7, 300, 70, 1),                          # must still decode
    ])
    with EventReader(p) as r:
        out = np.asarray(r.read_all())
    assert len(out) == 1, f"empty mask should yield only the trailing CD, got {len(out)}"
    assert (int(out["x"][0]), int(out["y"][0]), int(out["p"][0])) == (300, 70, 1)


def test_evt4_vector_and_scalar_mixed(tmp_path: Any) -> None:
    """Interleaved scalar CD, vector CD (both polarities) and TIME_HIGH decode
    in order with correct timestamps."""
    from evutils.io import EventReader
    p = _header_only(tmp_path)
    _append(p, [
        _time_high(10),
        _cd(10, 1, 2, 0),                              # scalar, p=0
        _cd_vec_base(10, 500, 300, 1), (1 << 1) | (1 << 2),  # x=501,502 p=1
        _cd_vec_base(10, 8, 9, 0),     (1 << 0),             # x=8 p=0
        _cd(10, 3, 4, 1),                              # scalar, p=1
    ])
    with EventReader(p) as r:
        out = np.asarray(r.read_all())
    assert np.array_equal(out["x"], [1, 501, 502, 8, 3])
    assert np.array_equal(out["y"], [2, 300, 300, 9, 4])
    assert np.array_equal(out["p"], [0, 1, 1, 0, 1])
    assert np.array_equal(out["t"], [10, 10, 10, 10, 10])


def test_evt4_vector_full_mask(tmp_path: Any) -> None:
    """A fully-set 32-bit mask emits 32 contiguous events base_x .. base_x+31."""
    from evutils.io import EventReader
    p = _header_only(tmp_path)
    base_x = 64
    _append(p, [_time_high(3), _cd_vec_base(3, base_x, 11, 1), 0xFFFFFFFF])
    with EventReader(p) as r:
        out = np.asarray(r.read_all())
    assert len(out) == 32
    assert np.array_equal(out["x"], np.arange(base_x, base_x + 32))
    assert np.all(out["y"] == 11) and np.all(out["p"] == 1) and np.all(out["t"] == 3)
