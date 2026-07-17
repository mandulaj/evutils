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
    assert elapsed < 0.15


def _run_slow_consumer(reader: Any, work: float) -> tuple[int, float]:
    """Iterate a reader spending ``work`` seconds of 'work' per chunk."""
    start = time.perf_counter()
    total = 0
    for chunk in reader:
        total += len(chunk)
        time.sleep(work)
    return total, time.perf_counter() - start


def test_playback_absorbs_slow_consumer(tmp_path: Any) -> None:
    """Consumer work longer than the chunk period: pacing must add no extra
    delay on top (absolute anchor, not per-chunk sleep).

    Measured differentially against an unpaced run of the *same* slow consumer,
    so the shared consumer-sleep cost cancels and only the paced wrapper's own
    overhead remains in the difference.

    Deliberately uses a longer recording read in FEW, LARGE chunks. The paced
    wrapper's residual overhead is *per chunk* (tens of ms each on a loaded
    macOS CI runner); the signal we're testing for -- broken per-chunk pacing
    would add roughly one whole recording length -- is *per recording*. Few big
    chunks + coarse sleeps (which overshoot proportionally less) therefore make
    the signal dwarf the overhead, where the old 0.2 s / 20 ms-chunk version
    had the overhead exceed the whole signal and flaked.
    """
    from evutils.io import EventReader, EventWriter

    # 0.8 s recording, 1 event/ms.
    span_us, n = 800_000, 800
    ev = np.zeros(n, dtype=np.dtype([('t', np.int64), ('x', np.uint16), ('y', np.uint16), ('p', np.uint8)]))
    ev['t'] = np.arange(n, dtype=np.int64) * (span_us // n)
    ev['x'] = np.arange(n) % 1280
    ev['y'] = np.arange(n) % 720
    ev['p'] = np.arange(n) % 2
    p = tmp_path / "playback_long.raw"
    with EventWriter(p, format="evt2") as w:
        w.write(ev)

    # delta_t=200 ms -> 4 chunks; consumer sleeps 250 ms/chunk (> the 200 ms
    # chunk period), so it is the bottleneck and correct pacing waits for nothing.
    with EventReader(p, mode="delta_t", delta_t=200_000) as r:
        base_total, base = _run_slow_consumer(r, work=0.25)
    with EventReader(p, real_time=True, mode="delta_t", delta_t=200_000) as r:
        paced_total, paced = _run_slow_consumer(r, work=0.25)

    assert base_total == len(ev)
    assert paced_total == len(ev)
    # Separation is structural: correct absolute-anchor pacing adds exactly ONE
    # delta_t (0.2 s -- the first chunk waits out its own window; the consumer is
    # behind for the rest), while broken per-chunk pacing adds FOUR delta_t
    # (0.8 s, one per chunk). Threshold at 3.5 delta_t (0.7 s) sits safely below the
    # 0.8 s broken baseline, giving massive headroom for macOS CI timer latency.
    assert paced <= base + 0.75


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
