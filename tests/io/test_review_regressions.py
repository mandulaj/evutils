"""Regression tests for correctness bugs found in review (July 2026).

Each test pins one confirmed bug so the fix stays in place:

1. Double-seek timestamp corruption -- a second ``seek()`` must not inherit the
   first seek's TIME_HIGH wrap correction (``_evt.py`` ``seek``).
2. Dense kernels scrambling coordinates -- structured arrays whose fields are
   stored in a non-canonical order must still decode ``(t, x, y, p)`` correctly
   (``jit.py`` ``lazy_njit_unwrapped_events``).
3. EVT3 timestamps overflowing at ~71.6 min -- both the scalar and the
   vectorised event paths must keep full 64-bit timestamps past 2**32 µs
   (``csrc/evt3.c`` state pipeline + ``EMIT_SOA``).
"""
import os
import tempfile
from pathlib import Path

import numpy as np
import pytest

from evutils.io import EventReader, EventWriter
from evutils.types import Event_dtype


# --------------------------------------------------------------------------- #
# Bug 1: double seek must not inherit the previous seek's wrap correction
# --------------------------------------------------------------------------- #
def _ramp(n=20_000, dt=1_000):
    ev = np.zeros(n, dtype=Event_dtype)
    ev["t"] = np.arange(n, dtype=np.int64) * dt
    ev["x"] = np.arange(n) % 1280
    ev["y"] = np.arange(n) % 720
    ev["p"] = np.arange(n) % 2
    return ev


@pytest.mark.parametrize("fmt", ["evt3", "evt2", "evt21"])
def test_double_seek_across_wrap_then_drain(tmp_path, fmt):
    """First seek lands past the EVT3 TIME_HIGH wrap (2**24 µs); the second seek
    to a pre-wrap target must then read a contiguous, correctly-timestamped tail.

    Before the fix the stale ``_seek_correction`` from the first seek turned the
    second seek's correction into ``W2 - W1``; the staged boundary chunk came out
    right but every chunk after it was offset by ``-W1``.
    """
    ev = _ramp()
    p = tmp_path / f"ramp.{fmt}.raw"
    with EventWriter(str(p), format=fmt) as w:
        w.write(ev)

    with EventReader(str(p), n_events=1000) as r:
        r.seek(t=18_000_000)          # past 2**24 = 16_777_216 (EVT3 wrap)
        T = 2_000_000                 # backward, pre-wrap
        exp = int(np.searchsorted(ev["t"], T))
        r.seek(t=T)
        got = []
        while True:
            c = r.read()
            if len(c) == 0:
                break
            got.append(np.asarray(c.t).copy())

    tail = np.concatenate(got)
    assert np.array_equal(tail, ev["t"][exp:])


# --------------------------------------------------------------------------- #
# Bug 2: dense kernels must not scramble coordinates on non-canonical field order
# --------------------------------------------------------------------------- #
def test_dense_kernel_independent_of_field_order():
    """A structured array stored as (x, y, t, p) must produce the same frame as
    the canonical (t, x, y, p) layout -- the kernel takes (t, x, y, p)
    positionally, so fields must be selected by name, not by dtype order.
    """
    from evutils.dense import frame_gray

    canon = np.array(
        [(5, 10, 20, 1), (7, 30, 40, 0)],
        dtype=[("t", "<i8"), ("x", "<u2"), ("y", "<u2"), ("p", "i1")],
    )
    scrambled = np.array(
        [(10, 20, 5, 1), (30, 40, 7, 0)],
        dtype=[("x", "<u2"), ("y", "<u2"), ("t", "<i8"), ("p", "i1")],
    )

    f_canon = frame_gray(canon, width=100, height=100)
    f_scram = frame_gray(scrambled, width=100, height=100)

    assert np.array_equal(f_canon, f_scram)
    assert f_canon[20, 10] == 255   # (y=20, x=10, p=1)
    assert f_canon[40, 30] == 0     # (y=40, x=30, p=0)


# --------------------------------------------------------------------------- #
# Bug 3: EVT3 timestamps must survive past the uint32 (~71.6 min) ceiling
# --------------------------------------------------------------------------- #
def _evt3_word(packet_type, data):
    return np.uint16(((packet_type & 0xF) << 12) | (data & 0x0FFF))


def _write_evt3_overflow_stream(path, wraps=257):
    """Craft a minimal EVT3 payload whose time base wraps ``wraps`` times, so the
    accumulator exceeds 2**32 µs, then emit one scalar and one vector event that
    share that timestamp. ``wraps=257`` gives ts = 257 * 2**24 + 100 > 2**32.
    """
    TH, TL, ADDR_Y, ADDR_X, VECT_BASE_X, VECT_12, OTHERS = 0x8, 0x6, 0x0, 0x2, 0x3, 0x4, 0xE
    words = []
    # Each (0xFFF, 0x000) TIME_HIGH pair makes the 24-bit field wrap once.
    for _ in range(wraps):
        words.append(_evt3_word(TH, 0xFFF))
        words.append(_evt3_word(TH, 0x000))
    words.append(_evt3_word(TL, 0x064))                 # ts = wraps*2**24 + 100
    words.append(_evt3_word(ADDR_Y, 5))
    words.append(_evt3_word(ADDR_X, 10 | (1 << 11)))    # scalar: x=10, p=1
    words.append(_evt3_word(VECT_BASE_X, 0x100 | (1 << 11)))
    words.append(_evt3_word(VECT_12, 0x001))            # one vector event at base
    words += [_evt3_word(OTHERS, 0)] * 8                # look-ahead / tail padding
    payload = np.array(words, dtype=np.uint16).tobytes()

    header = (b"% evt 3.0\n% format EVT3;height=720;width=1280\n"
              b"% geometry 1280x720\n% end\n")
    with open(path, "wb") as f:
        f.write(header)
        f.write(payload)
    return wraps * (1 << 24) + 100


def test_evt3_timestamp_past_uint32_ceiling(tmp_path):
    """Both the scalar (ADDR_X) and the vectorised (EMIT_SOA) event paths must
    keep the full 64-bit timestamp once the time base passes 2**32 µs.
    """
    p = tmp_path / "evt3_overflow.raw"
    expected = _write_evt3_overflow_stream(str(p))
    assert expected > 2**32

    with EventReader(str(p)) as r:
        ev = r.read_all()

    assert len(ev) == 2
    assert int(ev.t[0]) == expected   # scalar
    assert int(ev.t[1]) == expected   # vector -- would truncate to low 32 bits before the fix


# --------------------------------------------------------------------------- #
# Bug 4: corrupt packets -> robust warn+skip by default, raise under strict
# --------------------------------------------------------------------------- #
def _write_evt3_corrupt_stream(path):
    """EVT3 payload with a stray VECT_8 (a vector continuation with no preceding
    base) between two otherwise-valid scalar events. The decoder must skip the
    bad word, keeping both valid events.
    """
    TH, TL, ADDR_Y, ADDR_X, VECT_8, OTHERS = 0x8, 0x6, 0x0, 0x2, 0x5, 0xE
    words = [
        _evt3_word(TH, 0x000),
        _evt3_word(TL, 0x064),               # ts = 100
        _evt3_word(ADDR_Y, 5),
        _evt3_word(ADDR_X, 10 | (1 << 11)),  # valid event #1
        _evt3_word(VECT_8, 0x0AA),           # stray VECT_8 -> corrupt
        _evt3_word(ADDR_X, 20 | (1 << 11)),  # valid event #2
    ]
    words += [_evt3_word(OTHERS, 0)] * 8
    payload = np.array(words, dtype=np.uint16).tobytes()
    header = (b"% evt 3.0\n% format EVT3;height=720;width=1280\n"
              b"% geometry 1280x720\n% end\n")
    with open(path, "wb") as f:
        f.write(header)
        f.write(payload)


def test_corrupt_packet_robust_by_default(tmp_path):
    """Default decoder warns and skips the bad word, still returning both valid
    events (Metavision's UNRELIABLE behaviour) -- it must not raise."""
    import warnings

    p = tmp_path / "evt3_corrupt.raw"
    _write_evt3_corrupt_stream(str(p))

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with EventReader(str(p)) as r:
            ev = r.read_all()

    assert [int(v) for v in ev.x] == [10, 20]
    assert any("malformed" in str(w.message).lower() for w in caught)


def test_corrupt_packet_raises_in_strict_mode(tmp_path):
    """strict=True re-raises the malformed packet instead of skipping it
    (Metavision's SAFE behaviour)."""
    from evutils.io._evt import EventDecoder_EVT

    p = tmp_path / "evt3_corrupt.raw"
    _write_evt3_corrupt_stream(str(p))
    # Feed the bytes in-memory: a mid-decode raise on an mmap source pins the
    # payload view via the traceback, so the mmap close would mask the real
    # error with BufferError. An in-memory BufferSource has no such close hazard.
    raw = p.read_bytes()

    with pytest.raises(Exception, match="(?i)malformed|strict"):
        with EventReader(raw, file_decoder=EventDecoder_EVT, strict=True) as r:
            r.read_all()


# --------------------------------------------------------------------------- #
# Seek gaps: seek must not break trigger decoding; seek(n=) past EOF is empty
# --------------------------------------------------------------------------- #
def _evt3_trigger_words(t, tid, value):
    return np.array([
        0x8000 | ((t >> 12) & 0xFFF),          # TIME_HIGH
        0x6000 | (t & 0xFFF),                  # TIME_LOW
        0xA000 | ((tid & 0xF) << 8) | (value & 1),
    ], dtype=np.uint16)


def test_seek_preserves_triggers_evt3(tmp_path):
    """A time seek with ext_trigger=True must still decode the external triggers
    that follow the landing point (seek x triggers path)."""
    ev = np.zeros(100, dtype=Event_dtype)
    ev["t"] = np.arange(100, dtype=np.int64) * 100    # 0 .. 9900 µs
    ev["x"] = np.arange(100) % 1280
    ev["y"] = np.arange(100) % 720
    ev["p"] = np.arange(100) % 2
    triggers = [(10_000, 3, 1), (10_064, 3, 0), (20_000, 7, 1)]

    p = tmp_path / "seek_trig.raw"
    with EventWriter(str(p), format="evt3") as w:
        w.write(ev)
    words = np.concatenate([_evt3_trigger_words(t, i, v) for t, i, v in triggers])
    with open(p, "ab") as f:                          # append crafted trigger words
        f.write(words.tobytes())

    T = 5_000
    exp = int(np.searchsorted(ev["t"], T))
    ev_got, tr_got = [], []
    with EventReader(str(p), n_events=40, ext_trigger=True) as r:
        r.seek(t=T)
        while True:
            e, tr = r.read()
            if len(e) == 0 and len(tr) == 0:
                break
            if len(e):
                ev_got.append(np.asarray(e.t).copy())
            if len(tr):
                tr_got.append(np.asarray(tr.t).copy())

    ev_all = np.concatenate(ev_got)
    tr_all = np.concatenate(tr_got)
    assert int(ev_all[0]) == int(ev["t"][exp])        # events start at the target
    assert np.array_equal(np.sort(tr_all), [10_000, 10_064, 20_000])


def test_seek_by_event_index_past_eof(tmp_path):
    """seek(n=) beyond the last event lands at EOF and the next read is empty."""
    ev = np.zeros(1000, dtype=Event_dtype)
    ev["t"] = np.arange(1000, dtype=np.int64) * 1_000
    p = tmp_path / "seek_n_eof.raw"
    with EventWriter(str(p), format="evt3") as w:
        w.write(ev)

    with EventReader(str(p), n_events=100) as r:
        r.seek(n=10_000_000)
        assert len(r.read()) == 0


# --------------------------------------------------------------------------- #
# Bug 5 (round 3): the dedicated delta_t C parser must honour the post-seek
# TIME_HIGH wrap correction -- corrected end_ts in, corrected timestamps out.
# --------------------------------------------------------------------------- #
def test_parse_step_delta_t_applies_seek_correction(tmp_path):
    """After a seek whose bookmark lies past the EVT3 wrap (2**24 µs), the
    parser is reset and decodes in the raw (low) timeline; parse_step applies
    ``_seek_correction`` on the way out, and parse_step_delta_t must do the
    same -- translating the caller's absolute ``end_ts`` into the raw timeline
    for the C call and shifting the decoded slice back. Before the fix the
    window boundary was never reached and the emitted timestamps were low by
    one wrap period.
    """
    from evutils.io._evt import EventDecoder_EVT
    from evutils.io._native_core import (
        EVUTILS_PARSE_WINDOW_DONE, EventSoABuffers, TriggerSoABuffers,
        events_view,
    )
    from evutils.io._source import make_source

    n, dt = 200_000, 100                     # ts 0 .. 20_000_000 > 2**24
    ev = _ramp(n, dt)
    p = tmp_path / "wrap_dt.raw"
    with EventWriter(str(p), format="evt3") as w:
        w.write(ev)

    d = EventDecoder_EVT(make_source(str(p)), chunk_size=2048)
    d.init()
    T = 19_800_000                           # bookmark for this target sits past the wrap
    res, rem, _ = d.seek(t=T)
    assert d._seek_correction == 1 << 24     # one lost wrap to restore
    assert int(res.ts) == T

    # First event still in the stream (right after the returned remainder).
    next_ts = int(rem.t[-1]) + dt
    end_ts = next_ts + 50_000
    out = EventSoABuffers(65_536)
    tr = TriggerSoABuffers(1024)
    while True:
        out.c.capacity = out.capacity
        appended, status = d.parse_step_delta_t(out, tr, end_ts)
        if status == EVUTILS_PARSE_WINDOW_DONE or d.is_eof():
            break
    d.close()

    t = events_view(out).t
    assert len(t) == 500                     # exactly one 50 ms window of the ramp
    assert int(t[0]) == next_ts              # absolute (corrected) timeline
    assert int(t[-1]) < end_ts               # boundary honoured in that timeline
    assert bool(np.all(np.diff(t) == dt))


# --------------------------------------------------------------------------- #
# Bug 6 (round 3): index="metavision" must actually use the sidecar -- the
# normalize_ts pre-anchor used to force-build an in-memory index over it.
# --------------------------------------------------------------------------- #
def _write_mv_sidecar(raw_path):
    """Synthesize a Metavision ``.tmp_index`` sidecar for an EVT3 file, with
    bookmarks taken from a real decode pass (exact offsets/counts)."""
    from pathlib import Path

    from evutils.io._evt import EventDecoder_EVT
    from evutils.io._index import IncrementalSeekIndex, metavision_index_path
    from evutils.io._source import make_source

    d = EventDecoder_EVT(make_source(str(raw_path)))
    d.init()
    idx = IncrementalSeekIndex(
        words=d._words, start_offset=d._start_offset,
        input_cls=d._input_cls, parser_cls=type(d._parser),
        tail_pad=d._tail_pad, word_dtype=d._word_dtype, chunk_cap=2048,
        time_high=d._TIME_HIGH_TYPE[d._format], stride_words=512,
    )
    idx._build_until(target_t=2**62)         # build to EOF
    ts = np.asarray(idx._ts_list, dtype=np.int64)
    off = np.asarray(idx._off_list, dtype=np.int64)
    cum = np.asarray(idx._cum_list, dtype=np.int64)
    counts = np.diff(np.append(cum, idx._cum))
    word_size = np.dtype(d._word_dtype).itemsize
    byte_off = d._payload_off + off * word_size
    d.close()

    rec_dtype = np.dtype([("ts", "<i8"), ("byte_offset", "<u8"), ("count", "<u4")])
    recs = np.zeros(len(ts) + 1, dtype=rec_dtype)
    recs["ts"][:-1] = ts
    recs["byte_offset"][:-1] = byte_off
    recs["count"][:-1] = counts
    recs[-1] = (0x4D414749, 0, 0)            # trailing completeness marker (dropped)

    sidecar = metavision_index_path(str(raw_path))
    size = Path(raw_path).stat().st_size
    with open(sidecar, "wb") as f:
        f.write(f"% size {size}\n% ts_shift_us 0\n% end\n".encode())
        f.write(recs.tobytes())


def test_metavision_sidecar_index_is_used(tmp_path):
    """With index="metavision" a fresh sidecar must be the index actually
    consulted (StaticSeekIndex, not the in-memory build), and the seek must
    land exactly."""
    from evutils.io._index import StaticSeekIndex

    ev = _ramp()
    p = tmp_path / "sidecar.raw"
    with EventWriter(str(p), format="evt3") as w:
        w.write(ev)
    _write_mv_sidecar(p)

    T = 12_345_000
    exp = int(np.searchsorted(ev["t"], T))
    with EventReader(str(p), n_events=1000, index="metavision") as r:
        landed = r.seek(t=T)
        c = r.read()
        used = r._file_decoder._index
        assert isinstance(used, StaticSeekIndex)
        assert r._file_decoder._index_is_ours is False
    assert landed.ts == int(ev["t"][exp])
    assert int(c.t[0]) == int(ev["t"][exp])


@pytest.mark.parametrize("fmt", ["evt3", "evt2"])
def test_seek_multi_bookmark_exact(tmp_path, fmt):
    """Built-index seeks must land exactly on files large enough for MANY
    bookmarks. Regression for the bookmark-alignment bug: bookmarks recorded at
    arbitrary parse-step boundaries leave a reset parser without a time base
    (raw ts = low bits only), so the wrap-correction snap rounded to a whole
    spurious wrap period. Bookmarks are now TIME_HIGH-aligned.
    """
    n, dt = 200_000, 100                     # ~10 bookmarks at the default stride
    ev = _ramp(n, dt)
    p = tmp_path / f"multi_bm.{fmt}.raw"
    with EventWriter(str(p), format=fmt) as w:
        w.write(ev)

    with EventReader(str(p), n_events=2000) as r:
        for T in (3_000_000, 12_345_000, 19_800_000):
            exp = int(np.searchsorted(ev["t"], T))
            landed = r.seek(t=T)
            c = r.read()
            assert landed.ts == int(ev["t"][exp]), f"t={T}"
            assert int(c.t[0]) == int(ev["t"][exp])
        # index seek across bookmarks too
        r.seek(n=150_000)
        c = r.read()
        assert int(c.t[0]) == int(ev["t"][150_000])


def test_seek_then_read_normalize_ts_matches_read_then_seek(tmp_path):
    """normalize_ts must be call-order independent: seeking before the first
    read must normalize against the stream's first event, not against the
    landing point (or 0)."""
    ev = _ramp()
    ev["t"] += 100_000                        # stream starts at 100 ms
    p = tmp_path / "norm_order.raw"
    with EventWriter(str(p), format="evt3") as w:
        w.write(ev)

    T = 5_000_000
    with EventReader(str(p), n_events=3000, normalize_ts=True) as r:
        r.seek(t=T)                           # seek FIRST, no prior read
        a = r.read()
    with EventReader(str(p), n_events=3000, normalize_ts=True) as r:
        r.read()                              # anchor by reading first
        r.seek(t=T)
        b = r.read()

    assert int(a.t[0]) == int(b.t[0]) == T - 100_000


# --------------------------------------------------------------------------- #
# delta_t window dedup (_finalize_delta_t_window): the C path (EVT3) and the
# generic searchsorted path (DAT/EVT2) must still produce identical windows.
# --------------------------------------------------------------------------- #

def _delta_t_windows(path, dt, **kw):
    """Read one recording in delta_t mode, returning per-window (count, first,
    last) plus the concatenated timestamps."""
    counts, bounds, parts = [], [], []
    with EventReader(str(path), delta_t=dt, **kw) as r:
        while True:
            c = r.read()
            if len(c) == 0:
                break
            t = np.asarray(c.t)
            counts.append(len(c))
            bounds.append((int(t[0]), int(t[-1])))
            parts.append(t.copy())
    ts = np.concatenate(parts) if parts else np.empty(0, dtype=np.int64)
    return counts, bounds, ts


@pytest.mark.parametrize("fmt", ["evt3", "evt2", "dat"])
def test_delta_t_windows_lossless_and_bounded(tmp_path, fmt):
    """Concatenated delta_t windows equal a full read, and every window (bar a
    count-cut one) spans strictly less than delta_t. Exercises both the C
    (evt3) and generic (evt2/dat) window producers behind the shared finalize."""
    n, dt_step = 60_000, 137
    ev = _ramp(n, dt_step)
    if fmt == "dat":
        p = tmp_path / "win.dat"
        with EventWriter(str(p), width=1280, height=720) as w:
            w.write(ev)
    else:
        p = tmp_path / f"win_{fmt}.raw"
        with EventWriter(str(p), format=fmt) as w:
            w.write(ev)

    with EventReader(str(p), mode="all") as r:
        ref = np.asarray(r.read_all().t)

    dt = 200_000
    counts, bounds, ts = _delta_t_windows(p, dt)
    assert np.array_equal(ts, ref)
    assert sum(counts) == len(ref)
    assert all(hi < lo + dt for lo, hi in bounds)


@pytest.mark.parametrize("fmt", ["evt3", "dat"])
def test_delta_t_count_cut_lossless(tmp_path, fmt):
    """A tight n_events cap forces count-cut resumes; no events are lost and the
    output still equals a full read (count-cut resume path in the shared
    finalize)."""
    n, dt_step = 60_000, 137
    ev = _ramp(n, dt_step)
    if fmt == "dat":
        p = tmp_path / "cut.dat"
        with EventWriter(str(p), width=1280, height=720) as w:
            w.write(ev)
    else:
        p = tmp_path / f"cut_{fmt}.raw"
        with EventWriter(str(p), format=fmt) as w:
            w.write(ev)

    with EventReader(str(p), mode="all") as r:
        ref = np.asarray(r.read_all().t)

    _, _, ts = _delta_t_windows(p, 200_000, n_events=1500)
    assert np.array_equal(ts, ref)


_FAN = Path(__file__).resolve().parents[2] / "data" / "fan"


@pytest.mark.skipif(not (_FAN / "evt3_fan.raw").is_file(),
                    reason="data/fan/evt3_fan.raw fixture not present")
def test_delta_t_dedup_real_evt3():
    p = _FAN / "evt3_fan.raw"
    with EventReader(str(p)) as r:
        ref = np.asarray(r.read_all().t)
    counts, bounds, ts = _delta_t_windows(p, 20_000)
    assert np.array_equal(ts, ref)
    assert all(hi < lo + 20_000 for lo, hi in bounds)


@pytest.mark.skipif(not (_FAN / "dat_fan.dat").is_file(),
                    reason="data/fan/dat_fan.dat fixture not present")
def test_delta_t_dedup_real_dat():
    p = _FAN / "dat_fan.dat"
    with EventReader(str(p)) as r:
        ref = np.asarray(r.read_all().t)
    counts, bounds, ts = _delta_t_windows(p, 20_000)
    assert np.array_equal(ts, ref)
    assert all(hi < lo + 20_000 for lo, hi in bounds)
