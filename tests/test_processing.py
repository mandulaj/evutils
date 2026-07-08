import numpy as np
import pytest
from evutils.processing import mask_events, normalize_ts
from evutils.types import Event_dtype

def test_mask_events():
    events = np.array(
        [(100, 0, 0, 1), (200, 1, 1, 1), (300, 2, 2, 0)],
        dtype=Event_dtype
    )
    mask = np.array([
        [1, 0, 0],
        [0, 0, 0],
        [0, 0, 1]
    ])
    
    # Normal case
    masked = mask_events(events, mask)
    assert len(masked) == 2
    assert masked['x'].tolist() == [0, 2]

    # Empty events
    empty_events = np.array([], dtype=Event_dtype)
    assert len(mask_events(empty_events, mask)) == 0

    # ValueError: 1D mask
    with pytest.raises(ValueError, match="Mask must be a 2D array"):
        mask_events(events, np.array([1, 0, 1]))

    # ValueError: Out of bounds
    out_of_bounds = np.array([(100, 3, 3, 1)], dtype=Event_dtype)
    with pytest.raises(ValueError, match="Events x and y coordinates must be within the mask dimensions"):
        mask_events(out_of_bounds, mask)
        
    # ValueError: negative coordinates (using generic dtype since Event_dtype has unsigned x/y)
    # Event_dtype has x, y as u2 (unsigned 16-bit), so negative values wrap around to large positive.
    # Therefore, negative values are effectively out of bounds.

def test_normalize_ts():
    events = np.array(
        [(100, 0, 0, 1), (200, 1, 1, 1), (300, 2, 2, 0)],
        dtype=Event_dtype
    )
    
    # Normal case
    norm_events = normalize_ts(events.copy())
    assert norm_events['t'].tolist() == [0, 100, 200]

    # Start ts != 0
    norm_events_start = normalize_ts(events.copy(), start_ts=50)
    assert norm_events_start['t'].tolist() == [50, 150, 250]

    # Empty array
    empty_events = np.array([], dtype=Event_dtype)
    assert len(normalize_ts(empty_events)) == 0
