import numpy as np
import pytest

from evutils.types import Event_dtype, EventArray
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

@pytest.fixture(params=["aos", "soa"])
def event_format(request):
    return request.param

def create_events(data, format="aos"):
    events = np.zeros(len(data), dtype=Event_dtype)
    for i, (t, x, y, p) in enumerate(data):
        events[i]['t'] = t
        events[i]['x'] = x
        events[i]['y'] = y
        events[i]['p'] = p
    
    if format == "soa":
        return EventArray.from_aos(events)
    return events

def test_timesurface(event_format):
    events = create_events([(100, 10, 20, 1), (200, 15, 25, 0)], format=event_format)
    ts = timesurface(events, width=100, height=50, tau=100)
    assert ts.shape == (50, 100)
    assert ts[20, 10] == pytest.approx(np.exp(-(200 - 100) / 100.0))
    assert ts[25, 15] == pytest.approx(-1.0)

def test_timesurface_empty(event_format):
    events = create_events([], format=event_format)
    ts = timesurface(events, width=100, height=50)
    assert ts.shape == (50, 100)
    assert np.all(ts == 0)

def test_voxel_histogram(event_format):
    events = create_events([(100, 10, 20, 1), (150, 15, 25, 0), (200, 10, 20, 1)], format=event_format)
    vh = voxel_histogram(events, width=100, height=50, n_bins=2, dt=100)
    assert vh.shape == (2, 50, 100, 3)

def test_voxel_histogram_empty(event_format):
    events = create_events([], format=event_format)
    vh = voxel_histogram(events, width=100, height=50, n_bins=10, dt=1000)
    assert vh.shape == (10, 50, 100, 3)
    assert np.all(vh == 0)

def test_voxel_histogram_less_than_3_events(event_format):
    events = create_events([(100, 10, 20, 1), (150, 15, 25, 0)], format=event_format)
    vh = voxel_histogram(events, width=100, height=50, n_bins=2, dt=100)
    assert vh.shape == (2, 50, 100, 3)
    assert np.all(vh == 0)

def test_voxel_histogram_exceeds_dt(event_format):
    events = create_events([(100, 10, 20, 1), (150, 15, 25, 0), (250, 10, 20, 1)], format=event_format)
    with pytest.raises(ValueError):
        voxel_histogram(events, width=100, height=50, n_bins=2, dt=100)

def test_histogram(event_format):
    events = create_events([(100, 10, 20, 1), (200, 15, 25, 0), (250, 10, 20, 1)], format=event_format)
    h = histogram(events, width=100, height=50, fill=False)
    assert h.shape == (50, 100, 3)
    assert h[20, 10, 0] == 2 # Red channel for p=1
    assert h[25, 15, 2] == 1 # Blue channel for p=0

def test_histogram_empty(event_format):
    events = create_events([], format=event_format)
    h = histogram(events, width=100, height=50)
    assert h.shape == (50, 100, 3)
    assert np.all(h == 0)

def test_histogram_fill(event_format):
    events = create_events([(100, 10, 20, 1)], format=event_format)
    h = histogram(events, width=100, height=50, fill=True)
    assert h.shape == (50, 100, 3)
    assert h[20, 10, 0] == 255

def test_wedge_histogram(event_format):
    events = create_events([(100, 10, 20, 1), (200, 15, 25, 0)], format=event_format)
    wh = wedge_histogram(events, width=100, height=50, tl=300)
    assert wh.shape == (50, 100, 3)

def test_frame_diff(event_format):
    events = create_events([(100, 10, 20, 1), (150, 15, 25, 0), (200, 10, 20, 1)], format=event_format)
    fd = frame_diff(events, width=100, height=50)
    assert fd.shape == (50, 100)
    assert fd[20, 10] == 2
    assert fd[25, 15] == -1

def test_frame_diff_empty(event_format):
    events = create_events([], format=event_format)
    fd = frame_diff(events, width=100, height=50)
    assert fd.shape == (50, 100)
    assert np.all(fd == 0)

def test_frame_rgb(event_format):
    events = create_events([(100, 10, 20, 1), (150, 15, 25, 0)], format=event_format)
    frgb = frame_rgb(events, width=100, height=50)
    assert frgb.shape == (50, 100, 3)
    assert np.array_equal(frgb[20, 10], [255, 0, 0])
    assert np.array_equal(frgb[25, 15], [0, 0, 255])
    assert np.array_equal(frgb[0, 0], [0, 0, 0])

def test_frame_gray(event_format):
    events = create_events([(100, 10, 20, 1), (150, 15, 25, 0)], format=event_format)
    fg = frame_gray(events, width=100, height=50)
    assert fg.shape == (50, 100)
    assert fg[20, 10] == 255
    assert fg[25, 15] == 0
    assert fg[0, 0] == 128

def test_tore(event_format):
    events = create_events([(100, 10, 20, 1), (150, 15, 25, 0), (200, 10, 20, 1)], format=event_format)
    t = tore(events, width=100, height=50, n_events=2, tau=100)
    assert t.shape == (50, 100, 2, 2)
    assert t[20, 10, 1, 1] == pytest.approx(1.0)
    assert t[20, 10, 0, 1] == pytest.approx(np.exp(-1.0))

def test_tore_empty(event_format):
    events = create_events([], format=event_format)
    t = tore(events, width=100, height=50, n_events=4)
    assert t.shape == (50, 100, 4, 2)
    assert np.all(t == 0)

# --- saturation / overflow edge cases (branch coverage) --------------------

def test_histogram_saturation(event_format):
    events = create_events([(i, 10, 20, 1) for i in range(300)], format=event_format)
    h = histogram(events, width=100, height=50, fill=False)
    assert h[20, 10, 0] == 255

def test_wedge_histogram_saturation(event_format):
    events = create_events([(100, 10, 20, 1), (200, 10, 20, 1)], format=event_format)
    wh = wedge_histogram(events, width=100, height=50, tl=300)
    assert wh[20, 10, 0] == 255

def test_tore_fifo_overflow(event_format):
    events = create_events([(100, 10, 20, 1), (150, 10, 20, 1), (200, 10, 20, 1)], format=event_format)
    t = tore(events, width=100, height=50, n_events=2, tau=100)
    assert t.shape == (50, 100, 2, 2)
    assert t[20, 10, 1, 1] == pytest.approx(1.0)
    assert t[20, 10, 0, 1] == pytest.approx(np.exp(-0.5))

def test_voxel_histogram_more_windows_than_bins(event_format):
    events = create_events([(0, 10, 20, 1), (50, 11, 21, 0), (100, 12, 22, 1)], format=event_format)
    vh = voxel_histogram(events, width=100, height=50, n_bins=2, dt=100)
    assert vh.shape == (2, 50, 100, 3)

def test_voxel_histogram_fewer_windows_than_bins(event_format):
    events = create_events([(0, 10, 20, 1), (10, 11, 21, 0), (20, 12, 22, 1)], format=event_format)
    vh = voxel_histogram(events, width=100, height=50, n_bins=10, dt=10000)
    assert vh.shape == (10, 50, 100, 3)
    assert np.all(vh[1:] == 0)
