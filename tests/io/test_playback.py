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


def test_playback_absorbs_slow_consumer(recording: Any) -> None:
    """Consumer work longer than the chunk period: pacing must add no extra
    delay on top (absolute anchor, not per-chunk sleep)."""
    from evutils.io import EventReader
    p, ev = recording
    start = time.perf_counter()
    total = 0
    with EventReader(p, real_time=True, mode="delta_t", delta_t=20_000) as r:
        for chunk in r:
            total += len(chunk)
            time.sleep(0.04)  # 40 ms work per 20 ms chunk -> consumer-bound
    elapsed = time.perf_counter() - start
    assert total == len(ev)
    # 10 chunks x 40 ms work = 0.4 s consumer time; pacing is already behind
    # schedule, so total must stay near the work time, not work + recording.
    assert elapsed < 0.55


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
