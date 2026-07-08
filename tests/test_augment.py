import numpy as np
import pytest
from evutils.augment import drop_random_events
from evutils.types import Event_dtype

def test_drop_random_events():
    events = np.array(
        [(i, i, i*100, 1) for i in range(100)],
        dtype=Event_dtype
    )

    # Normal case
    dropped = drop_random_events(events, drop_rate=0.1)
    assert len(dropped) == 90

    # Value errors
    with pytest.raises(ValueError, match="drop_rate must be between 0 and 1"):
        drop_random_events(events, drop_rate=0.0)

    with pytest.raises(ValueError, match="drop_rate must be between 0 and 1"):
        drop_random_events(events, drop_rate=1.0)
        
    with pytest.raises(ValueError, match="drop_rate must be between 0 and 1"):
        drop_random_events(events, drop_rate=-0.1)
        
    with pytest.raises(ValueError, match="drop_rate must be between 0 and 1"):
        drop_random_events(events, drop_rate=1.5)

    # Empty array
    empty_events = np.array([], dtype=Event_dtype)
    assert len(drop_random_events(empty_events, drop_rate=0.5)) == 0
