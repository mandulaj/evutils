"""A/B parity: ``evutils.io.v2.EventReader`` vs the V1 monolith.

The V2 reader is a decomposed reimplementation of V1 (strategies + cursor +
pacer over a shared ReadContext). These tests pin it to V1 event-for-event
across the read-mode / config matrix, the seek surface, and the batch / async
delivery paths, on synthetic files (all seekable formats) plus the real
``data/fan`` recordings when present (bounded to keep CI fast).
"""
from pathlib import Path

import numpy as np
import pytest

from evutils.io import EventReader as V1, EventWriter
from evutils.io.v2 import EventReader as V2
from evutils.types import DataBatch, Event_dtype

_DATA = Path(__file__).resolve().parents[2] / "data" / "fan"

N = 30_000
DT = 1_000  # max t = 30_000_000 > 2**24: EVT wrap correction is exercised.

# Seekable formats writable via the extension / format dispatch.
FORMATS = [
    ("evt3", "raw"), ("evt4", "raw"), ("evt2", "raw"), ("evt21", "raw"),
    ("dat", "dat"), ("csv", "csv"), ("npz", "npz"),
]


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _ev(x):
    if isinstance(x, DataBatch):
        return x.events
    return x[0] if isinstance(x, tuple) else x


def _tr(x):
    if isinstance(x, DataBatch):
        return x.triggers
    return x[1] if isinstance(x, tuple) else None


def assert_window_eq(a, b, label):
    ea, eb = _ev(a), _ev(b)
    assert len(ea) == len(eb), f"{label}: len {len(ea)} != {len(eb)}"
    for f in ("x", "y", "p", "t"):
        assert np.array_equal(getattr(ea, f), getattr(eb, f)), f"{label}: field {f}"
    ta, tb = _tr(a), _tr(b)
    la = 0 if ta is None else len(ta)
    lb = 0 if tb is None else len(tb)
    assert la == lb, f"{label}: trigger len {la} != {lb}"
    if la:
        for f in ("t", "p", "id"):
            assert np.array_equal(getattr(ta, f), getattr(tb, f)), f"{label}: trig {f}"


def assert_iter_eq(path, label, max_windows=None, decoder=None, **kw):
    extra = {"file_decoder": decoder} if decoder is not None else {}
    r1 = V1(path, **kw, **extra)
    r2 = V2(path, **kw, **extra)
    it1, it2 = iter(r1), iter(r2)
    i = 0
    while max_windows is None or i < max_windows:
        a = next(it1, None)
        b = next(it2, None)
        if a is None and b is None:
            break
        assert (a is None) == (b is None), f"{label}: window {i} count mismatch"
        assert_window_eq(a, b, f"{label}[win {i}]")
        i += 1
    r1.close(); r2.close()
    return i


def _ramp(n=N, dt=DT, t0=0):
    ev = np.zeros(n, dtype=Event_dtype)
    ev["t"] = t0 + np.arange(n, dtype=np.int64) * dt
    ev["x"] = np.arange(n) % 1280
    ev["y"] = np.arange(n) % 720
    ev["p"] = np.arange(n) % 2
    return ev


def _write(path, fmt, ev):
    if fmt in ("evt3", "evt4", "evt2", "evt21"):
        with EventWriter(path, format=fmt) as w:
            w.write(ev)
    else:
        with EventWriter(path, width=1280, height=720) as w:
            w.write(ev)


@pytest.fixture(scope="module")
def synth(tmp_path_factory):
    """One ramp recording per seekable format (module-scoped)."""
    d = tmp_path_factory.mktemp("v2parity")
    ev = _ramp()
    paths = {}
    for fmt, ext in FORMATS:
        p = d / f"ramp_{fmt}.{ext}"
        _write(p, fmt, ev)
        paths[fmt] = p
    return paths, ev


# --------------------------------------------------------------------------- #
# Read-mode / config matrix (exhaustive on synthetic files)
# --------------------------------------------------------------------------- #
CONFIGS = [
    ("delta_t", dict(delta_t=1_000_000)),
    ("delta_t_small", dict(delta_t=300_000)),
    ("n_events", dict(n_events=4000)),
    ("mixed", dict(delta_t=1_000_000, n_events=4000)),
    ("all", dict(mode="all")),
    ("delta_t_norm", dict(delta_t=1_000_000, normalize_ts=True)),
    ("n_events_norm", dict(n_events=4000, normalize_ts=True)),
    ("all_norm", dict(mode="all", normalize_ts=True)),
    ("delta_t_reuse", dict(delta_t=1_000_000, reuse_buffers=True)),
    ("delta_t_start", dict(delta_t=1_000_000, normalize_ts=True, start_ts=5000)),
    ("mixed_norm", dict(delta_t=1_000_000, n_events=4000, normalize_ts=True)),
]


@pytest.mark.parametrize("fmt", [f[0] for f in FORMATS])
@pytest.mark.parametrize("cname", [c[0] for c in CONFIGS])
def test_read_mode_parity(synth, fmt, cname):
    paths, _ = synth
    kw = dict(CONFIGS)[cname]
    n = assert_iter_eq(paths[fmt], f"{fmt}/{cname}", **kw)
    assert n > 0


@pytest.mark.parametrize("fmt", [f[0] for f in FORMATS])
def test_read_all_parity(synth, fmt):
    paths, _ = synth
    for kw in ({}, dict(normalize_ts=True), dict(delta_t=1_000_000)):
        r1 = V1(paths[fmt], **kw); r2 = V2(paths[fmt], **kw)
        assert_window_eq(r1.read_all(), r2.read_all(), f"{fmt}/read_all/{kw}")
        r1.close(); r2.close()


# --------------------------------------------------------------------------- #
# Seek parity (result tuple + landing read)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("fmt", [f[0] for f in FORMATS])
@pytest.mark.parametrize("norm", [False, True], ids=["abs", "norm"])
def test_seek_parity(synth, fmt, norm):
    paths, ev = synth
    kw = dict(delta_t=1_000_000, normalize_ts=norm)
    scenarios = [
        ("t_mid", ("t", 7_000_000)),
        ("t_early", ("t", 1_234_000)),
        ("n_mid", ("n", 12_000)),
        ("n_zero", ("n", 0)),
        ("t_oob", ("t", 10**15)),
        ("n_oob", ("n", 10**9)),
    ]
    for name, (axis, val) in scenarios:
        r1 = V1(paths[fmt], **kw); r2 = V2(paths[fmt], **kw)
        a = r1.seek(**{axis: val}); b = r2.seek(**{axis: val})
        assert a == b, f"{fmt}/{name}: SeekResult {a} != {b}"
        assert r1.event_index == r2.event_index
        assert r1.last_seek == r2.last_seek
        assert_window_eq(r1.read(), r2.read(), f"{fmt}/{name}")
        r1.close(); r2.close()


@pytest.mark.parametrize("fmt", [f[0] for f in FORMATS])
def test_seek_backward_relative_parity(synth, fmt):
    paths, _ = synth
    for name, seq in [
        ("backward", [("n", 15000, False), ("n", 300, False)]),
        ("relative_t", [("t", 5_000_000, False), ("t", 5_000_000, True)]),
        ("relative_n", [("n", 5000, False), ("n", 2000, True)]),
    ]:
        r1 = V1(paths[fmt], delta_t=1_000_000); r2 = V2(paths[fmt], delta_t=1_000_000)
        a = b = None
        for axis, val, rel in seq:
            a = r1.seek(**{axis: val}, relative=rel)
            b = r2.seek(**{axis: val}, relative=rel)
        assert a == b, f"{fmt}/{name}: {a} != {b}"
        assert_window_eq(r1.read(), r2.read(), f"{fmt}/{name}")
        r1.close(); r2.close()


def test_seek_linear_fallback_parity(synth):
    """Non-seekable source: linear iterate-and-drop must match V1."""
    from evutils.io._csv import EventDecoder_Csv

    class _NonSeekable:
        def __init__(self, data): self._d = data; self._p = 0
        def read(self, size=-1):
            if size < 0: size = len(self._d) - self._p
            c = self._d[self._p:self._p + size]; self._p += len(c); return c
        def readable(self): return True
        def seekable(self): return False
        def seek(self, *a):
            import io
            raise io.UnsupportedOperation("not seekable")
        def tell(self): return self._p

    paths, _ = synth
    data = Path(paths["csv"]).read_bytes()
    T = 6_000_000
    r1 = V1(_NonSeekable(data), n_events=3000, file_decoder=EventDecoder_Csv)
    r2 = V2(_NonSeekable(data), n_events=3000, file_decoder=EventDecoder_Csv)
    a = r1.seek(t=T); b = r2.seek(t=T)
    assert a == b, f"nonseekable: {a} != {b}"
    assert_window_eq(r1.read(), r2.read(), "nonseekable")
    r1.close(); r2.close()


# --------------------------------------------------------------------------- #
# Delivery paths: batch_mode + async_read
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("fmt", ["evt3", "dat", "csv"])
def test_batch_mode_parity(synth, fmt):
    paths, _ = synth
    assert_iter_eq(paths[fmt], f"{fmt}/batch", batch_mode=True, delta_t=1_000_000)
    r = V2(paths[fmt], batch_mode=True, delta_t=1_000_000)
    assert isinstance(r.read(), DataBatch)
    for w in r:
        assert isinstance(w, DataBatch)
    r.close()
    r = V2(paths[fmt], batch_mode=True)
    assert isinstance(r.read_all(), DataBatch)
    r.close()


@pytest.mark.parametrize("fmt", ["evt3", "evt2", "dat"])
@pytest.mark.parametrize("cfg", [
    dict(mode="delta_t", delta_t=1_000_000),
    dict(mode="n_events", n_events=4000),
], ids=["delta_t", "n_events"])
def test_async_read_parity(synth, fmt, cfg):
    paths, _ = synth
    assert_iter_eq(paths[fmt], f"{fmt}/async", async_read=True, **cfg)
    assert_iter_eq(paths[fmt], f"{fmt}/async6", async_read=True, prefetch_depth=6, **cfg)


# --------------------------------------------------------------------------- #
# Trigger parity (EVT3 supports external triggers)
# --------------------------------------------------------------------------- #
def test_ext_trigger_parity():
    """External triggers must slice identically. Uses the real EVT3 fan
    recording (the EVT encoders do not emit trigger packets), so skipped when
    the data fixture is absent."""
    path = _DATA / "evt3_fan.raw"
    if not path.is_file():
        pytest.skip(f"{path} not present")
    for kw in (dict(delta_t=10_000), dict(delta_t=10_000, n_events=50_000)):
        assert_iter_eq(path, f"ext_trig/{kw}", max_windows=120, ext_trigger=True, **kw)


# --------------------------------------------------------------------------- #
# Real recordings (bounded; skipped when data/fan is absent)
# --------------------------------------------------------------------------- #
_FAN = [
    ("evt3", _DATA / "evt3_fan.raw"),
    ("evt2", _DATA / "evt2_fan.raw"),
    ("dat", _DATA / "dat_fan.dat"),
]


@pytest.mark.parametrize("fmt,path", _FAN, ids=[f[0] for f in _FAN])
@pytest.mark.parametrize("cname", ["delta_t", "n_events", "mixed", "delta_t_norm",
                                   "delta_t_reuse"])
def test_real_fan_parity(fmt, path, cname):
    if not path.is_file():
        pytest.skip(f"{path} not present")
    kw = {
        "delta_t": dict(delta_t=10_000),
        "n_events": dict(n_events=50_000),
        "mixed": dict(delta_t=10_000, n_events=50_000),
        "delta_t_norm": dict(delta_t=10_000, normalize_ts=True),
        "delta_t_reuse": dict(delta_t=10_000, reuse_buffers=True),
    }[cname]
    n = assert_iter_eq(path, f"{fmt}/{cname}", max_windows=120, **kw)
    assert n == 120


@pytest.mark.parametrize("fmt,path", _FAN, ids=[f[0] for f in _FAN])
def test_real_fan_seek_parity(fmt, path):
    if not path.is_file():
        pytest.skip(f"{path} not present")
    for axis, val in [("n", 100_000), ("t", 50_000)]:
        r1 = V1(path, delta_t=10_000); r2 = V2(path, delta_t=10_000)
        a = r1.seek(**{axis: val}); b = r2.seek(**{axis: val})
        assert a == b, f"{fmt}/seek {axis}={val}: {a} != {b}"
        assert_window_eq(r1.read(), r2.read(), f"{fmt}/seek")
        r1.close(); r2.close()
