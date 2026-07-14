import numpy as np
import pytest

from evutils.types import Event_dtype
from evutils.repr import (
    timesurface,
    voxel_histogram,
    histogram,
    wedge_histogram,
    frame_diff,
    frame_rgb,
    frame_gray,
    tore
)

def create_events(data):
    events = np.zeros(len(data), dtype=Event_dtype)
    for i, (t, x, y, p) in enumerate(data):
        events[i]['t'] = t
        events[i]['x'] = x
        events[i]['y'] = y
        events[i]['p'] = p
    return events

def test_timesurface():
    events = create_events([(100, 10, 20, 1), (200, 15, 25, 0)])
    ts = timesurface(events, width=100, height=50, tau=100)
    assert ts.shape == (50, 100)
    assert ts[20, 10] == pytest.approx(np.exp(-(200 - 100) / 100.0))
    assert ts[25, 15] == pytest.approx(-1.0)

def test_timesurface_empty():
    events = create_events([])
    ts = timesurface(events, width=100, height=50)
    assert ts.shape == (50, 100)
    assert np.all(ts == 0)

def test_voxel_histogram():
    events = create_events([(100, 10, 20, 1), (150, 15, 25, 0), (200, 10, 20, 1)])
    vh = voxel_histogram(events, width=100, height=50, n_bins=2, dt=100)
    assert vh.shape == (2, 50, 100, 3)

def test_voxel_histogram_empty():
    events = create_events([])
    vh = voxel_histogram(events, width=100, height=50, n_bins=10, dt=1000)
    assert vh.shape == (10, 50, 100, 3)
    assert np.all(vh == 0)

def test_voxel_histogram_less_than_3_events():
    events = create_events([(100, 10, 20, 1), (150, 15, 25, 0)])
    vh = voxel_histogram(events, width=100, height=50, n_bins=2, dt=100)
    assert vh.shape == (2, 50, 100, 3)
    assert np.all(vh == 0)

def test_voxel_histogram_exceeds_dt():
    events = create_events([(100, 10, 20, 1), (150, 15, 25, 0), (250, 10, 20, 1)])
    with pytest.raises(AssertionError):
        voxel_histogram(events, width=100, height=50, n_bins=2, dt=100)

def test_histogram():
    events = create_events([(100, 10, 20, 1), (200, 15, 25, 0), (250, 10, 20, 1)])
    h = histogram(events, width=100, height=50, fill=False)
    assert h.shape == (50, 100, 3)
    assert h[20, 10, 0] == 2 # Red channel for p=1
    assert h[25, 15, 2] == 1 # Blue channel for p=0

def test_histogram_empty():
    events = create_events([])
    h = histogram(events, width=100, height=50)
    assert h.shape == (50, 100, 3)
    assert np.all(h == 0)

def test_histogram_fill():
    events = create_events([(100, 10, 20, 1)])
    h = histogram(events, width=100, height=50, fill=True)
    assert h.shape == (50, 100, 3)
    assert h[20, 10, 0] == 255

def test_wedge_histogram():
    events = create_events([(100, 10, 20, 1), (200, 15, 25, 0)])
    wh = wedge_histogram(events, width=100, height=50, tl=300)
    assert wh.shape == (50, 100, 3)

def test_frame_diff():
    events = create_events([(100, 10, 20, 1), (150, 15, 25, 0), (200, 10, 20, 1)])
    fd = frame_diff(events, width=100, height=50)
    assert fd.shape == (50, 100)
    assert fd[20, 10] == 2
    assert fd[25, 15] == -1

def test_frame_diff_empty():
    events = create_events([])
    fd = frame_diff(events, width=100, height=50)
    assert fd.shape == (50, 100)
    assert np.all(fd == 0)

def test_frame_rgb():
    events = create_events([(100, 10, 20, 1), (150, 15, 25, 0)])
    frgb = frame_rgb(events, width=100, height=50)
    assert frgb.shape == (50, 100, 3)
    assert np.array_equal(frgb[20, 10], [255, 0, 0])
    assert np.array_equal(frgb[25, 15], [0, 0, 255])
    assert np.array_equal(frgb[0, 0], [0, 0, 0])

def test_frame_gray():
    events = create_events([(100, 10, 20, 1), (150, 15, 25, 0)])
    fg = frame_gray(events, width=100, height=50)
    assert fg.shape == (50, 100)
    assert fg[20, 10] == 255
    assert fg[25, 15] == 0
    assert fg[0, 0] == 128

def test_tore():
    events = create_events([(100, 10, 20, 1), (150, 15, 25, 0), (200, 10, 20, 1)])
    t = tore(events, width=100, height=50, n_events=2, tau=100)
    assert t.shape == (50, 100, 2, 2)
    assert t[20, 10, 1, 1] == pytest.approx(1.0)
    assert t[20, 10, 0, 1] == pytest.approx(np.exp(-1.0))

def test_tore_empty():
    events = create_events([])
    t = tore(events, width=100, height=50, n_events=4)
    assert t.shape == (50, 100, 4, 2)
    assert np.all(t == 0)


# --- saturation / overflow edge cases (branch coverage) --------------------

def test_histogram_saturation():
    """A uint8 histogram bin clips at 255 instead of overflowing."""
    events = create_events([(i, 10, 20, 1) for i in range(300)])  # 300 hits, one pixel
    h = histogram(events, width=100, height=50, fill=False)
    assert h[20, 10, 0] == 255  # capped, not wrapped to 300 % 256


def test_wedge_histogram_saturation():
    """A wedge bin saturates: a second selected hit on the same pixel is not
    added once the bin is already at 255."""
    events = create_events([(100, 10, 20, 1), (200, 10, 20, 1)])
    wh = wedge_histogram(events, width=100, height=50, tl=300)
    assert wh[20, 10, 0] == 255  # single 255, second hit skipped


def test_tore_fifo_overflow():
    """With more events than n_events on one pixel/polarity, the oldest fall out
    of the FIFO; only the most recent n_events survive."""
    events = create_events([(100, 10, 20, 1), (150, 10, 20, 1), (200, 10, 20, 1)])
    t = tore(events, width=100, height=50, n_events=2, tau=100)
    assert t.shape == (50, 100, 2, 2)
    # newest (t=200) at slot 1, second-newest (t=150) at slot 0; t=100 dropped.
    assert t[20, 10, 1, 1] == pytest.approx(1.0)
    assert t[20, 10, 0, 1] == pytest.approx(np.exp(-0.5))


def test_voxel_histogram_more_windows_than_bins():
    """Extra time windows beyond n_bins are dropped (the loop breaks). Span
    0..100 at bin_dt=50 yields 3 windows, one more than n_bins=2."""
    events = create_events([(0, 10, 20, 1), (50, 11, 21, 0), (100, 12, 22, 1)])
    vh = voxel_histogram(events, width=100, height=50, n_bins=2, dt=100)
    assert vh.shape == (2, 50, 100, 3)


def test_voxel_histogram_fewer_windows_than_bins():
    """Fewer time windows than n_bins: the loop exhausts normally (no break),
    leaving the trailing bins zero. Span 20 at bin_dt=1000 is a single window."""
    events = create_events([(0, 10, 20, 1), (10, 11, 21, 0), (20, 12, 22, 1)])
    vh = voxel_histogram(events, width=100, height=50, n_bins=10, dt=10000)
    assert vh.shape == (10, 50, 100, 3)
    assert np.all(vh[1:] == 0)  # only the first window populated
