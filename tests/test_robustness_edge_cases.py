import pytest
import numpy as np
from evutils.types import EventArray, TriggerArray
from evutils.types import EventsChecker
from evutils.chunking import window_delta_t, sliding_window, stream_n_events, stream_delta_t
from evutils.filtering import mask_events
from evutils.transforms.functional import normalize_ts
from evutils.transforms import drop_random_events
from evutils.vis.plot3d import plot_3d, plot_3d_timesurface

# --- 1. Chunking Edge Cases ---

def test_chunking_zero_delta_t_raises():
    events = EventArray(t=[1, 2, 3], x=[0, 0, 0], y=[0, 0, 0], p=[0, 0, 0])
    with pytest.raises(ValueError, match="delta_t must be positive"):
        next(window_delta_t(events, delta_t=0))

    with pytest.raises(ValueError, match="delta_t and window_size must be positive"):
        next(sliding_window(events, delta_t=0, window_size=10))

def test_stream_n_events_zero_raises():
    def dummy_stream():
        yield EventArray(t=[1], x=[0], y=[0], p=[0])
        
    with pytest.raises(ValueError, match="n_events must be positive"):
        next(stream_n_events(dummy_stream(), n_events=0))

def test_stream_delta_t_trigger_only_stream():
    # Stream with only triggers, no events
    def trigger_stream():
        for i in range(1, 4):
            ev = EventArray(t=[], x=[], y=[], p=[])
            tr = TriggerArray(t=[i * 1000], p=[1], id=[1])
            yield ev, tr

    chunks = list(stream_delta_t(trigger_stream(), delta_t=1000))
    # It should yield chunks based on trigger timestamps
    assert len(chunks) > 0
    # Last chunk should have triggers
    assert len(chunks[-1][1]) > 0


# --- 2. Vis Empty Array Edge Cases ---

def test_vis_empty_arrays():
    import matplotlib.pyplot as plt
    empty_events = EventArray.empty()
    
    # These should not crash
    fig, ax = plot_3d(empty_events)
    assert ax is not None
    plt.close(fig)
    
    fig, ax = plot_3d_timesurface(empty_events)
    assert ax is not None
    plt.close(fig)
    
    # Open3D
    pytest.importorskip("open3d")
    from evutils.vis.open3d import o3d_draw_events
    o3d_draw_events(empty_events)


# --- 3. Types and Utils Edge Cases ---

def test_event_array_length_mismatch():
    with pytest.raises(ValueError, match="Length mismatch"):
        EventArray(t=[1, 2], x=[1], y=[1, 2], p=[1, 2])

def test_trigger_array_length_mismatch():
    with pytest.raises(ValueError, match="Length mismatch"):
        TriggerArray(t=[1, 2], p=[1], id=[1, 2])

def test_event_array_flattening():
    # Multidimensional inputs should be flattened gracefully
    events = EventArray(t=[[1, 2]], x=[[1, 2]], y=[[1, 2]], p=[[0, 1]])
    assert len(events) == 2
    assert events.t.ndim == 1

def test_events_checker_invalid_type():
    with pytest.raises(TypeError, match="events must be a NumPy array or SoaArray"):
        EventsChecker([1, 2, 3])

def test_empty_list_indexing():
    events = EventArray(t=[1], x=[1], y=[1], p=[1])
    with pytest.raises(ValueError, match="empty list of fields"):
        events[[]]


# --- 4. Processing and Augment Edge Cases ---

def test_drop_random_events_nan():
    events = EventArray(t=[1, 2], x=[1, 2], y=[1, 2], p=[0, 1]).to_numpy()
    with pytest.raises(ValueError, match="drop_rate must be between 0 and 1"):
        drop_random_events(events, drop_rate=np.nan)

def test_mask_events_negative_coordinates():
    # If using a custom signed dtype
    custom_dtype = np.dtype([('t', np.int64), ('x', np.int16), ('y', np.int16), ('p', np.uint8)])
    events = np.array([(100, -5, 10, 1), (200, 10, 10, 1)], dtype=custom_dtype)
    mask = np.ones((20, 20), dtype=np.uint8)
    
    with pytest.raises(ValueError, match="non-negative"):
        mask_events(events, mask)

def test_normalize_ts_readonly():
    events = EventArray(t=[100, 200], x=[1, 2], y=[1, 2], p=[0, 1]).to_numpy()
    events.flags.writeable = False  # Simulate read-only mmap view
    
    # Should not crash, but return a normalized copy
    norm_events = normalize_ts(events)
    assert norm_events['t'][0] == 0
    assert norm_events['t'][1] == 100
    # Original is untouched
    assert events['t'][0] == 100

def test_normalize_ts_unstructured():
    events = np.array([[100, 1, 1, 1]])
    with pytest.raises(TypeError, match="Unsupported event format"):
        normalize_ts(events)
