"""EventAccumulator tests (evutils.io.buffer).

The accumulator is the staging buffer behind the streaming pipeline generators.
The stream tests only push small data, so the capacity-pressure paths -- rotate
(reclaim consumed front), trigger-buffer grow, and the overflow guard -- are
covered here directly.
"""
import numpy as np
import pytest

from evutils.types import EventArray, TriggerArray
from evutils.io.buffer import EventAccumulator


def ev(ts) -> EventArray:
    ts = list(ts)
    n = len(ts)
    return EventArray(t=ts, x=[0] * n, y=[0] * n, p=[0] * n)


def tr(ts) -> TriggerArray:
    ts = list(ts)
    n = len(ts)
    return TriggerArray(t=ts, p=[0] * n, id=list(range(n)))


def test_accumulator_slice_copy_advances():
    acc = EventAccumulator(capacity=100)
    acc.append(ev([0, 1, 2, 3]))
    assert len(acc) == 4
    e, _ = acc.slice_copy(2, 0)
    assert e.t.tolist() == [0, 1]
    assert len(acc) == 2
    assert acc.t_window().tolist() == [2, 3]


def test_accumulator_slice_copy_is_independent():
    acc = EventAccumulator(capacity=100)
    acc.append(ev([5, 6, 7]))
    e, _ = acc.slice_copy(3, 0)
    acc.reset()
    acc.append(ev([99, 99, 99]))
    assert e.t.tolist() == [5, 6, 7]  # earlier copy untouched by later writes


def test_accumulator_rotate_events_and_triggers():
    """Consuming the front then appending past the tail room triggers _rotate,
    which must preserve the unconsumed remainder of both events and triggers."""
    acc = EventAccumulator(capacity=64)  # trigger cap = 64 // 16 = 4
    acc.append(ev(range(8)), tr([0, 3, 6]))
    acc.slice_copy(5, 2)                  # consume 5 events, 2 triggers
    assert acc.t_window().tolist() == [5, 6, 7]

    acc.append(ev(range(100, 160)), tr([200, 201]))  # 64-8 < 60 -> rotate
    assert len(acc) == 63                            # 3 remainder + 60 new
    assert acc.t_window()[:3].tolist() == [5, 6, 7]
    assert acc.t_window()[3:].tolist() == list(range(100, 160))
    assert acc.t_window_tr().tolist() == [6, 200, 201]


def test_accumulator_rotate_after_full_consume():
    """Rotate when everything is consumed: the remainder is zero, so _rotate
    resets the offsets without copying (the n==0 / n_tr==0 branches)."""
    acc = EventAccumulator(capacity=10)
    acc.append(ev(range(6)), tr([1, 2]))
    acc.slice_copy(6, 2)                       # consume all events and triggers
    assert len(acc) == 0
    acc.append(ev(range(100, 108)))           # 10-6 < 8 -> rotate, nothing to move
    assert acc.t_window().tolist() == list(range(100, 108))


def test_accumulator_overflow_raises():
    """Appending more than the total capacity (even after a rotate) errors."""
    acc = EventAccumulator(capacity=5)
    acc.append(ev(range(4)))
    with pytest.raises(ValueError, match="full"):
        acc.append(ev(range(4)))


def test_accumulator_trigger_buffer_grows():
    """More triggers than the trigger buffer's capacity trigger a grow()."""
    acc = EventAccumulator(capacity=32)  # trigger cap = 32 // 16 = 2
    acc.append(ev([0]), tr([1, 2, 3, 4, 5]))  # 5 > 2 -> grow
    assert acc.t_window_tr().tolist() == [1, 2, 3, 4, 5]


def test_accumulator_prepare_rotates_for_headroom():
    acc = EventAccumulator(capacity=10)
    acc.append(ev(range(8)))
    acc.slice_copy(5, 0)              # start=5, 3 unconsumed
    b, _ = acc.prepare(5)             # 10-8 < 5 -> rotate
    assert acc.t_window().tolist() == [5, 6, 7]
    assert b.c.capacity == 8         # min(capacity, size + step) = min(10, 3+5)


def test_accumulator_reset_clears():
    acc = EventAccumulator(capacity=16)
    acc.append(ev([1, 2, 3]), tr([1]))
    acc.reset()
    assert len(acc) == 0
    assert acc.t_window().tolist() == []
    assert acc.t_window_tr().tolist() == []
