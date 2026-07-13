"""Real-time playback pacing tests (EventReader(real_time=True)).

Timing assertions are deliberately loose (lower bounds tight, upper bounds
generous) so scheduler jitter on CI machines does not flake them.
"""
import time

import numpy as np
import pytest

from typing import Any

SPAN_US = 200_000  # recording length: 0.2 s
N = 200            # one event per millisecond


@pytest.fixture
def recording(tmp_path: Any) -> Any:
    from evutils.io import EventWriter
    ev = np.zeros(N, dtype=np.dtype([('t', np.int64), ('x', np.uint16), ('y', np.uint16), ('p', np.uint8)]))
    ev['t'] = np.arange(N, dtype=np.int64) * (SPAN_US // N)
    ev['x'] = np.arange(N) % 1280
    ev['y'] = np.arange(N) % 720
    ev['p'] = np.arange(N) % 2
    p = tmp_path / "playback.raw"
    with EventWriter(p, format="evt2") as w:
        w.write(ev)
    return p, ev


def _drain(reader: Any) -> tuple[int, float]:
    start = time.perf_counter()
    total = 0
    for chunk in reader:
        total += len(chunk)
    return total, time.perf_counter() - start


def test_playback_paces_iteration_to_recording_speed(recording: Any) -> None:
    from evutils.io import EventReader
    p, ev = recording
    with EventReader(p, real_time=True, mode="delta_t", delta_t=20_000) as r:
        total, elapsed = _drain(r)
    assert total == len(ev)
    # 0.2 s recording: must take at least ~the recording duration, and not
    # wildly more (each chunk waits only up to its own timestamp).
    assert 0.15 <= elapsed < 1.0


def test_playback_speed_multiplier(recording: Any) -> None:
    from evutils.io import EventReader
    p, ev = recording
    with EventReader(p, real_time=True, playback_speed=4.0, mode="delta_t", delta_t=20_000) as r:
        total, elapsed = _drain(r)
    assert total == len(ev)
    # 0.2 s recording at 4x -> ~0.05 s
    assert 0.03 <= elapsed < 0.15


def test_playback_off_is_fast(recording: Any) -> None:
    from evutils.io import EventReader
    p, _ = recording
    with EventReader(p, mode="delta_t", delta_t=20_000) as r:
        _, elapsed = _drain(r)
    assert elapsed < 0.05


def _run_slow_consumer(reader: Any) -> tuple[int, float]:
    """Iterate a reader spending 40 ms of 'work' per chunk (consumer-bound:
    slower than the 20 ms chunk period)."""
    start = time.perf_counter()
    total = 0
    for chunk in reader:
        total += len(chunk)
        time.sleep(0.04)
    return total, time.perf_counter() - start


def test_playback_absorbs_slow_consumer(recording: Any) -> None:
    """Consumer work longer than the chunk period: pacing must add no extra
    delay on top (absolute anchor, not per-chunk sleep).

    Measured differentially against an unpaced run of the *same* slow consumer.
    Both runs pay identical sleep/scheduler jitter, so their difference cancels
    it -- unlike a fixed upper bound, which can't survive macOS CI sleep
    overshoot (the signal, one recording-length, is smaller than the jitter).
    Broken per-chunk pacing would still add ~the recording duration on top.
    """
    from evutils.io import EventReader
    p, ev = recording

    with EventReader(p, mode="delta_t", delta_t=20_000) as r:
        base_total, base = _run_slow_consumer(r)
    with EventReader(p, real_time=True, mode="delta_t", delta_t=20_000) as r:
        paced_total, paced = _run_slow_consumer(r)

    assert base_total == len(ev)
    assert paced_total == len(ev)
    # Consumer already runs behind schedule, so pacing waits for nothing: the
    # paced run must not exceed the unpaced one by more than half the 0.2 s
    # recording. Broken per-chunk sleeping would add the full recording.
    assert paced <= base + 0.1


def test_playback_paces_read_calls(recording: Any) -> None:
    from evutils.io import EventReader
    p, _ = recording
    with EventReader(p, real_time=True, mode="delta_t", delta_t=50_000) as r:
        start = time.perf_counter()
        r.read()  # anchor chunk: waits out its own 50 ms span
        r.read()
        elapsed = time.perf_counter() - start
    assert elapsed >= 0.075  # two 50 ms windows on the playback clock


def test_playback_with_async_read(recording: Any) -> None:
    from evutils.io import EventReader
    p, ev = recording
    with EventReader(p, real_time=True, async_read=True, mode="delta_t", delta_t=20_000) as r:
        total, elapsed = _drain(r)
    assert total == len(ev)
    assert 0.15 <= elapsed < 1.0


def test_playback_early_break_releases_prefetch(recording: Any) -> None:
    from evutils.io import EventReader
    p, _ = recording
    with EventReader(p, real_time=True, async_read=True, mode="delta_t", delta_t=20_000) as r:
        for chunk in r:
            break
        # the paced wrapper must close the prefetch iterator on break,
        # releasing the decoder for direct reads again
        r.reset()
        out = r.read_all()
    assert len(out) == N


def test_playback_reset_rearms_anchor(recording: Any) -> None:
    from evutils.io import EventReader
    p, _ = recording
    with EventReader(p, real_time=True, playback_speed=10.0, mode="delta_t", delta_t=100_000) as r:
        _drain(r)
        r.reset()
        start = time.perf_counter()
        total, _ = _drain(r)
        elapsed = time.perf_counter() - start
    assert total == N
    # second pass paces again from a fresh anchor (0.2 s / 10x = 20 ms)
    assert elapsed >= 0.015


def test_playback_invalid_speed_rejected(recording: Any) -> None:
    from evutils.io import EventReader
    p, _ = recording
    with pytest.raises(ValueError):
        EventReader(p, real_time=True, playback_speed=0)
    with pytest.raises(ValueError):
        EventReader(p, real_time=True, playback_speed=-1.5)


# --------------------------------------------------------------------------- #
# Idle-gap handling (max_gap)
# --------------------------------------------------------------------------- #
GAP_US = 500_000  # 0.5 s of silence before the action starts


@pytest.fixture
def gapped_recording(tmp_path: Any) -> Any:
    """A tiny blip at t=0, then 0.5 s of silence, then a short burst.

    Reproduces recordings whose content starts well after t=0 -- without
    gap-skipping, real-time pacing stalls on one long sleep to reach the burst.
    """
    from evutils.io import EventWriter
    t = np.concatenate([
        np.array([0], dtype=np.int64),                          # lead-in blip
        GAP_US + np.arange(50, dtype=np.int64) * 1000,          # burst: 50 ms
    ])
    ev = np.zeros(len(t), dtype=np.dtype([('t', np.int64), ('x', np.uint16), ('y', np.uint16), ('p', np.uint8)]))
    ev['t'] = t
    ev['x'] = np.arange(len(t)) % 1280
    ev['y'] = np.arange(len(t)) % 720
    ev['p'] = np.arange(len(t)) % 2
    p = tmp_path / "gapped.raw"
    with EventWriter(p, format="evt2") as w:
        w.write(ev)
    return p, ev


def test_max_gap_skips_dead_air(gapped_recording: Any) -> None:
    """With max_gap set, the 0.5 s silent lead-in is skipped, not slept through."""
    from evutils.io import EventReader
    p, ev = gapped_recording
    with EventReader(p, real_time=True, mode="delta_t", delta_t=20_000, max_gap=0.1) as r:
        total, elapsed = _drain(r)
    assert total == len(ev)
    # Only the ~50 ms burst is paced; the 0.5 s gap must not be waited out.
    assert elapsed < 0.3


def test_max_gap_none_is_strict_realtime(gapped_recording: Any) -> None:
    """max_gap=None keeps strict pacing: the full 0.5 s gap is honoured."""
    from evutils.io import EventReader
    p, ev = gapped_recording
    with EventReader(p, real_time=True, mode="delta_t", delta_t=20_000, max_gap=None) as r:
        total, elapsed = _drain(r)
    assert total == len(ev)
    # Must sleep through the ~0.5 s gap before the burst plays.
    assert elapsed >= 0.45


def test_max_gap_invalid_rejected(recording: Any) -> None:
    from evutils.io import EventReader
    p, _ = recording
    with pytest.raises(ValueError):
        EventReader(p, real_time=True, max_gap=0)
    with pytest.raises(ValueError):
        EventReader(p, real_time=True, max_gap=-1.0)
