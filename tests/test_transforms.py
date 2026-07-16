import numpy as np
import pytest
from evutils.transforms import drop_random_events, Compose, DropRandomEvents
from evutils.types import Event_dtype, EventArray

def test_drop_random_events():
    events = np.array(
        [(i, i, i*100, 1) for i in range(100)],
        dtype=Event_dtype
    )

    # Normal case
    dropped = drop_random_events(events, drop_rate=0.1)
    assert 70 <= len(dropped) <= 100

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


def test_compose_and_transforms():
    events = EventArray(
        t=np.arange(100),
        x=np.arange(100),
        y=np.arange(100),
        p=np.zeros(100, dtype=np.uint8)
    )
    
    # 1. Test single transform standalone
    transform = DropRandomEvents(drop_rate=0.1)
    dropped = transform(events)
    # The new JIT drop rate is binomial, so it won't be exactly 90
    assert 70 <= len(dropped) <= 100
    assert isinstance(dropped, EventArray)
    
    # 2. Test compose with multiple transforms
    pipeline = Compose([
        DropRandomEvents(drop_rate=0.1),
        DropRandomEvents(drop_rate=0.1)
    ])
    
    # Check execution plan
    assert len(pipeline._execution_plan) == 1
    assert pipeline._execution_plan[0][0] == "jit"
    assert len(pipeline._execution_plan[0][1]) == 2
    
    dropped_twice = pipeline(events)
    assert 60 <= len(dropped_twice) <= 100
    assert isinstance(dropped_twice, EventArray)
    
    # 3. Test interop with standard callables
    def dummy_transform(evs):
        # A non-JIT transform
        return evs
        
    pipeline_mixed = Compose([
        DropRandomEvents(drop_rate=0.1),
        dummy_transform,
        DropRandomEvents(drop_rate=0.1)
    ])
    
    # Execution plan should be jit -> standard -> jit
    assert len(pipeline_mixed._execution_plan) == 3
    assert pipeline_mixed._execution_plan[0][0] == "jit"
    assert pipeline_mixed._execution_plan[1][0] == "standard"
    assert pipeline_mixed._execution_plan[2][0] == "jit"
    
    dropped_mixed = pipeline_mixed(events)
    assert 60 <= len(dropped_mixed) <= 100
    assert isinstance(dropped_mixed, EventArray)
