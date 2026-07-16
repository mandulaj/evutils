"""Streaming API tests: EventStreamer (evutils.io.stream) and the pipeline
generators in evutils.chunking (stream_delta_t / stream_n_events /
stream_skip_to_time / stream_async / stream_paced_playback).

EventStreamer is exercised over a real encoder-written file; the pipeline
generators are driven with hand-built EventArray/TriggerArray chunks so every
branch (trigger vs no-trigger, boundary slicing, leftover flush, empty chunks,
exception propagation) is covered without needing a device.
"""
import time

import numpy as np
import pytest

from evutils.types import EventArray, TriggerArray, Event_dtype
from evutils.chunking import (
    stream_delta_t,
    stream_n_events,
    stream_skip_to_time,
    stream_async,
    stream_paced_playback,
)


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #

def ev(ts, xs=None, ys=None, ps=None) -> EventArray:
    ts = list(ts)
    n = len(ts)
    return EventArray(
        t=ts,
        x=xs if xs is not None else [0] * n,
        y=ys if ys is not None else [0] * n,
        p=ps if ps is not None else [0] * n,
    )


def tr(ts, ids=None, ps=None) -> TriggerArray:
    ts = list(ts)
    n = len(ts)
    return TriggerArray(t=ts, p=ps if ps is not None else [0] * n, id=ids if ids is not None else [0] * n)


def cat_t(chunks) -> list:
    """Flatten a list of chunks (or (ev, tr) tuples) into their event timestamps."""
    out = []
    for c in chunks:
        e = c[0] if isinstance(c, tuple) else c
        out.extend(e.t.tolist())
    return out


def cat_tr_t(chunks) -> list:
    out = []
    for c in chunks:
        out.extend(c[1].t.tolist())
    return out


# --------------------------------------------------------------------------- #
# stream_delta_t
# --------------------------------------------------------------------------- #

def test_stream_delta_t_regroups_across_chunk_boundaries():
    raw = [ev([0, 10, 20]), ev([30, 35, 55]), ev([60, 90, 95])]
    out = list(stream_delta_t(iter(raw), delta_t=30))
    # A trigger-less stream yields bare EventArrays (the output mirrors the
    # input shape). Every input event survives, in order.
    assert all(not isinstance(w, tuple) for w in out)
    assert cat_t(out) == [0, 10, 20, 30, 35, 55, 60, 90, 95]
    # full (non-final) windows respect the delta_t span
    for e in out[:-1]:
        if len(e) > 1:
            assert int(e.t[-1]) - int(e.t[0]) < 30


def test_stream_delta_t_with_triggers():
    raw = [
        (ev([0, 10]), tr([5], ids=[1])),
        (ev([40, 50]), tr([45], ids=[2])),
    ]
    out = list(stream_delta_t(iter(raw), delta_t=30))
    assert all(isinstance(c, tuple) for c in out)
    assert cat_t(out) == [0, 10, 40, 50]
    assert cat_tr_t(out) == [5, 45]


def test_stream_delta_t_empty_stream():
    assert list(stream_delta_t(iter([]), delta_t=30)) == []


def test_stream_delta_t_skips_empty_leading_chunks():
    raw = [ev([]), ev([1, 2, 3])]
    out = list(stream_delta_t(iter(raw), delta_t=100))
    assert cat_t(out) == [1, 2, 3]


# --------------------------------------------------------------------------- #
# stream_n_events
# --------------------------------------------------------------------------- #

def test_stream_n_events_fixed_size_with_remainder():
    raw = [ev(range(0, 5)), ev(range(5, 10)), ev(range(10, 13))]
    out = list(stream_n_events(iter(raw), n_events=4))
    # trigger-less input -> bare EventArray windows (output mirrors input shape)
    assert all(not isinstance(w, tuple) for w in out)
    assert [len(w) for w in out] == [4, 4, 4, 1]  # 13 = 3*4 + 1
    assert cat_t(out) == list(range(13))


def test_stream_n_events_with_triggers_exact_boundary():
    # total events an exact multiple of n_events -> exact-boundary trigger branch
    raw = [(ev([0, 1, 2, 3]), tr([1, 3], ids=[1, 2]))]
    out = list(stream_n_events(iter(raw), n_events=4))
    assert len(out) == 1
    e, t = out[0]
    assert e.t.tolist() == [0, 1, 2, 3]
    assert t.t.tolist() == [1, 3]


def test_stream_n_events_with_triggers_searchsorted_branch():
    # more events than n_events in the buffer -> searchsorted trigger split
    raw = [(ev([0, 1, 2, 3, 4, 5]), tr([0, 2, 5], ids=[1, 2, 3]))]
    out = list(stream_n_events(iter(raw), n_events=4))
    assert cat_t(out) == [0, 1, 2, 3, 4, 5]
    # triggers before t of the 4th event go with the first window
    assert cat_tr_t(out) == [0, 2, 5]


def test_stream_n_events_empty_stream():
    assert list(stream_n_events(iter([]), n_events=4)) == []


# --------------------------------------------------------------------------- #
# stream_skip_to_time
# --------------------------------------------------------------------------- #

def test_stream_skip_to_time_drops_whole_and_partial_chunks():
    raw = [ev([0, 10, 20]), ev([30, 40, 50])]
    out = list(stream_skip_to_time(iter(raw), start_ts=25))
    assert cat_t(out) == [30, 40, 50]  # first chunk dropped whole


def test_stream_skip_to_time_partial_slice_within_chunk():
    raw = [ev([0, 10, 20]), ev([30, 40])]
    out = list(stream_skip_to_time(iter(raw), start_ts=15))
    assert cat_t(out) == [20, 30, 40]  # first chunk sliced at 15


def test_stream_skip_to_time_with_triggers():
    raw = [
        (ev([0, 10, 20]), tr([5, 18], ids=[1, 2])),
        (ev([30, 40]), tr([35], ids=[3])),
    ]
    out = list(stream_skip_to_time(iter(raw), start_ts=15))
    assert cat_t(out) == [20, 30, 40]
    assert cat_tr_t(out) == [18, 35]  # trigger at 5 dropped


def test_stream_skip_to_time_never_reached_yields_nothing():
    raw = [ev([0, 10]), ev([20, 30])]
    out = list(stream_skip_to_time(iter(raw), start_ts=1000))
    assert out == []


# --------------------------------------------------------------------------- #
# stream_async
# --------------------------------------------------------------------------- #

def test_stream_async_preserves_order():
    raw = [ev([0, 1]), ev([2, 3]), ev([4])]
    out = list(stream_async(iter(raw)))
    assert cat_t(out) == [0, 1, 2, 3, 4]


def test_stream_async_with_triggers_and_none():
    raw = [(ev([0, 1]), tr([0], ids=[1])), (ev([2]), None)]
    out = list(stream_async(iter(raw)))
    assert cat_t(out) == [0, 1, 2]
    assert out[0][1].t.tolist() == [0]
    assert out[1][1] is None


def test_stream_async_copies_chunks():
    """Chunks are copied before crossing the thread, so mutating the source
    afterwards must not change what was yielded."""
    src = ev([7, 8])
    out = list(stream_async(iter([src])))
    src.t[0] = 999
    assert out[0].t.tolist() == [7, 8]


def test_stream_async_propagates_exceptions():
    def boom():
        yield ev([0])
        raise ValueError("upstream failed")

    with pytest.raises(ValueError, match="upstream failed"):
        list(stream_async(boom()))


# --------------------------------------------------------------------------- #
# stream_paced_playback
# --------------------------------------------------------------------------- #

def test_stream_paced_playback_preserves_data_fast():
    raw = [ev([0, 100]), ev([200, 300])]
    out = list(stream_paced_playback(iter(raw), playback_speed=1e6))  # ~no sleep
    assert cat_t(out) == [0, 100, 200, 300]


def test_stream_paced_playback_empty_chunk_passthrough():
    raw = [ev([]), ev([0, 10])]
    out = list(stream_paced_playback(iter(raw), playback_speed=1e6))
    assert cat_t(out) == [0, 10]


def test_stream_paced_playback_actually_waits():
    # 20 ms of stream at 2x speed -> ~10 ms of real wait (the sleep branch).
    raw = [ev([0]), ev([20_000])]
    start = time.perf_counter()
    out = list(stream_paced_playback(iter(raw), playback_speed=2.0))
    elapsed = time.perf_counter() - start
    assert cat_t(out) == [0, 20_000]
    assert elapsed >= 0.005  # some real delay occurred


# --------------------------------------------------------------------------- #
# EventStreamer (integration over a written file)
# --------------------------------------------------------------------------- #

def _make_events(n=500):
    a = np.zeros(n, dtype=Event_dtype)
    a['t'] = np.arange(n, dtype=np.int64) * 100
    a['x'] = np.arange(n) % 1280
    a['y'] = np.arange(n) % 720
    a['p'] = np.arange(n) % 2
    return a


def test_event_streamer_yields_all_events(tmp_path):
    from evutils.io import EventWriter, EventStreamer
    events = _make_events()
    p = tmp_path / "stream.raw"
    with EventWriter(p, format="evt3") as w:
        w.write(events)

    # EventStreamer yields chunks that alias a reused parser buffer, so retain a
    # copy per iteration (the documented low-level contract; stream_async does
    # the same copy).
    ts = []
    n = 0
    for c in EventStreamer(p):
        ts.append(c.t.copy())
        n += 1
    assert n > 0
    got = np.concatenate(ts)
    assert np.array_equal(got, events['t'])


def test_event_streamer_pipes_into_stream_n_events(tmp_path):
    from evutils.io import EventWriter, EventStreamer
    events = _make_events()
    p = tmp_path / "stream.raw"
    with EventWriter(p, format="evt3") as w:
        w.write(events)

    # stream_n_events copies into its accumulator, so piping the aliasing
    # EventStreamer through it is safe. Trigger-less input -> bare windows.
    windows = list(stream_n_events(EventStreamer(p), n_events=128))
    assert [len(w) for w in windows[:-1]] == [128] * (len(windows) - 1)
    got = np.concatenate([w.t for w in windows])
    assert np.array_equal(got, events['t'])


class _FakeDecoder:
    """Minimal decoder that yields one plain (non-tuple) event chunk then stops,
    used to drive EventStreamer with an explicit decoder_cls and to exercise the
    ext_trigger path when the decoder returns a bare EventArray."""
    def __init__(self, source, read_external_triggers=False, **kwargs):
        self._chunks = iter([ev([1, 2, 3]), ev([])])

    def init(self):
        pass

    def read_chunk(self):
        return next(self._chunks)


def test_event_streamer_explicit_decoder_cls_and_bare_chunk(tmp_path):
    from evutils.io import EventStreamer
    p = tmp_path / "any.raw"
    p.write_bytes(b"% end\n")  # make_source needs a real source; content unused

    # ext_trigger=True but the decoder returns a bare EventArray -> streamer must
    # pair it with an empty TriggerArray.
    out = list(EventStreamer(p, ext_trigger=True, decoder_cls=_FakeDecoder))
    assert len(out) == 1
    e, t = out[0]
    assert e.t.tolist() == [1, 2, 3]
    assert len(t) == 0

    # no-trigger path with the same explicit decoder_cls
    out2 = list(EventStreamer(p, decoder_cls=_FakeDecoder))
    assert [c.t.tolist() for c in out2] == [[1, 2, 3]]


def test_event_streamer_with_triggers(tmp_path):
    from evutils.io import EventWriter, EventStreamer
    events = _make_events()
    p = tmp_path / "stream.raw"
    with EventWriter(p, format="evt3") as w:
        w.write(events)

    ts, total_tr, all_tuples = [], 0, True
    for c in EventStreamer(p, ext_trigger=True):
        all_tuples = all_tuples and isinstance(c, tuple)
        ts.append(c[0].t.copy())   # copy: chunk aliases the reused parser buffer
        total_tr += len(c[1])
    assert all_tuples
    got = np.concatenate(ts)
    assert np.array_equal(got, events['t'])
    assert total_tr == 0  # no trigger words in the file
