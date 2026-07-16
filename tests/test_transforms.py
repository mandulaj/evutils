import numpy as np
import pytest
from evutils.transforms import (
    drop_random_events,
    Compose,
    DropEvent,
    DropRandomEvents,
    DropEventByTime,
    RandomFlipLR,
    SpatialJitter,
    TimeSkew,
    TimeNormalize,
    TimeJitter,
    RefractoryPeriod,
)
from evutils.transforms.functional import normalize_ts
from evutils.types import Event_dtype, EventArray


def test_normalize_ts_functional():
    events = np.array(
        [(100, 0, 0, 1), (200, 1, 1, 1), (300, 2, 2, 0)],
        dtype=Event_dtype
    )

    # Normal case -- earliest event shifts to 0; input untouched (non-mutating).
    norm = normalize_ts(events)
    assert norm['t'].tolist() == [0, 100, 200]
    assert events['t'].tolist() == [100, 200, 300]

    # start_ts != 0
    assert normalize_ts(events, start_ts=50)['t'].tolist() == [50, 150, 250]

    # Empty array
    assert len(normalize_ts(np.array([], dtype=Event_dtype))) == 0

    # EventArray in -> EventArray out
    ea = EventArray(t=[100, 200], x=[1, 2], y=[1, 2], p=[0, 1])
    out = normalize_ts(ea)
    assert isinstance(out, EventArray)
    assert out.t.tolist() == [0, 100]


def test_time_normalize_transform():
    ea = EventArray(t=[100, 200, 300], x=[0, 1, 2], y=[0, 1, 2], p=[1, 1, 0])
    assert TimeNormalize()(ea).t.tolist() == [0, 100, 200]
    assert TimeNormalize(start_ts=10)(ea).t.tolist() == [10, 110, 210]
    # Composable alongside other time transforms.
    out = Compose([TimeNormalize(), TimeSkew(coefficient=1.0, offset=0)])(ea)
    assert out.t.tolist() == [0, 100, 200]

def test_drop_random_events():
    events = np.array(
        [(i, i, i*100, 1) for i in range(100)],
        dtype=Event_dtype
    )

    # Normal case
    dropped = drop_random_events(events, drop_rate=0.1)
    assert 70 <= len(dropped) <= 100
    # Survivors must stay in temporal order
    assert np.all(np.diff(dropped['t']) >= 0)

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
    transform = DropEvent(p=0.1)
    dropped = transform(events)
    # The JIT drop rate is binomial, so it won't be exactly 90
    assert 70 <= len(dropped) <= 100
    assert isinstance(dropped, EventArray)

    # 2. Test compose with multiple transforms
    pipeline = Compose([
        DropEvent(p=0.1),
        DropEvent(p=0.1)
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
        DropEvent(p=0.1),
        dummy_transform,
        DropEvent(p=0.1)
    ])
    
    # Execution plan should be jit -> standard -> jit
    assert len(pipeline_mixed._execution_plan) == 3
    assert pipeline_mixed._execution_plan[0][0] == "jit"
    assert pipeline_mixed._execution_plan[1][0] == "standard"
    assert pipeline_mixed._execution_plan[2][0] == "jit"
    
    dropped_mixed = pipeline_mixed(events)
    assert 60 <= len(dropped_mixed) <= 100
    assert isinstance(dropped_mixed, EventArray)

def test_target_transformation():
    from evutils.transforms import Transform
    
    # Custom transform that modifies target
    class DummyTargetCrop(Transform):
        def _forward_jit(self, t, x, y, p):
            return t, x, y, p
            
        def _transform_target(self, target):
            if isinstance(target, dict) and "bbox" in target:
                target["bbox"] = [v - 10 for v in target["bbox"]]
            return target
            
    events = EventArray(t=[1], x=[1], y=[1], p=[1])
    target = {"class": "car", "bbox": [50, 50, 100, 100]}
    
    # 1. Standalone
    transform = DummyTargetCrop()
    out_events, out_target = transform(events, target=target.copy())
    assert out_target["bbox"] == [40, 40, 90, 90]
    
    # 2. Compose
    pipeline = Compose([
        DropEvent(p=0.1),
        DummyTargetCrop()
    ])
    
    out_events, out_target = pipeline(events, target=target.copy())
    assert out_target["bbox"] == [40, 40, 90, 90]


# --------------------------------------------------------------------------- #
# New tonic-style transforms
# --------------------------------------------------------------------------- #

def _make_events(n=200, w=64, h=48):
    """Deterministic-ish structured event array spread across a small sensor."""
    return np.array(
        [(i * 10, i % w, i % h, i % 2) for i in range(n)],
        dtype=Event_dtype,
    )


def test_drop_event_rename_and_validation():
    # Alias points at the same class.
    assert DropRandomEvents is DropEvent

    events = _make_events()
    dropped = DropEvent(p=0.2)(events)
    assert 130 <= len(dropped) <= 200
    # p=0 is a no-op (tonic-compatible), not an error.
    assert len(DropEvent(p=0.0)(events)) == len(events)

    for bad in (1.0, 1.5, -0.1, np.nan):
        with pytest.raises(ValueError, match=r"p must be in"):
            DropEvent(p=bad)

    # Tuple range samples a valid probability.
    ranged = DropEvent(p=(0.1, 0.3))(events)
    assert 100 <= len(ranged) <= 200


def test_random_flip_lr():
    events = EventArray(t=[0, 1, 2], x=[0, 10, 63], y=[1, 2, 3], p=[0, 1, 0])
    flipped = RandomFlipLR(sensor_size=(64, 48, 2), p=1.0)(events)
    assert list(flipped.x) == [63, 53, 0]         # width - 1 - x
    assert list(flipped.y) == [1, 2, 3]           # y untouched
    assert flipped.x.dtype == np.uint16
    # p=0 never flips.
    assert list(RandomFlipLR(sensor_size=(64, 48, 2), p=0.0)(events).x) == [0, 10, 63]


def test_time_skew():
    events = EventArray(t=[0, 100, 200], x=[1, 2, 3], y=[1, 2, 3], p=[1, 0, 1])
    out = TimeSkew(coefficient=2.0, offset=10)(events)
    assert list(out.t) == [10, 210, 410]
    assert out.t.dtype == np.int64


def test_time_jitter_clip_and_sort():
    events = EventArray(
        t=np.arange(500) * 100,
        x=np.zeros(500), y=np.zeros(500), p=np.zeros(500),
    )
    out = TimeJitter(std=50.0, clip_negative=True, sort_timestamps=True)(events)
    assert np.all(out.t >= 0)
    assert np.all(np.diff(out.t) >= 0)      # sorted


def test_spatial_jitter_clip_keeps_in_bounds():
    events = EventArray(
        t=np.arange(1000),
        x=np.full(1000, 32), y=np.full(1000, 24), p=np.zeros(1000),
    )
    out = SpatialJitter(sensor_size=(64, 48, 2), var_x=25.0, var_y=25.0,
                        clip_outliers=True)(events)
    assert np.all(out.x < 64) and np.all(out.y < 48)
    assert len(out) <= 1000


def test_refractory_period():
    # Two pixels. Pixel (0,0) fires at 20,25,200; pixel (1,1) at 21,22.
    events = EventArray(
        t=[20, 21, 22, 25, 200],
        x=[0, 1, 1, 0, 0],
        y=[0, 1, 1, 0, 0],
        p=[1, 1, 1, 1, 1],
    )
    out = RefractoryPeriod(delta=10)(events)
    # (0,0): keep t=20 (first), drop t=25 (gap 5<=10), keep t=200 (gap 175>10).
    # (1,1): keep t=21 (first), drop t=22 (gap 1<=10).
    assert sorted(out.t.tolist()) == [20, 21, 200]


def test_drop_by_time_removes_a_window():
    events = _make_events(n=300)
    out = DropEventByTime(duration_ratio=0.3)(events)
    assert len(out) < len(events)
    # duration_ratio=0 is a no-op.
    assert len(DropEventByTime(duration_ratio=0.0)(events)) == len(events)


def test_ndarray_and_eventarray_dispatch():
    """Same transform works on structured ndarray and EventArray alike."""
    aos = _make_events(n=50)
    soa = EventArray.from_aos(aos)
    skew = TimeSkew(coefficient=3.0)
    assert isinstance(skew(aos), np.ndarray)
    assert isinstance(skew(soa), EventArray)
    np.testing.assert_array_equal(skew(aos)["t"], skew(soa).t)


def test_metadata_propagates_through_transforms():
    events = EventArray(t=[0, 1, 2], x=[0, 10, 63], y=[1, 2, 3], p=[0, 1, 0],
                        metadata={"sensor_size": (64, 48)})
    # Single transform keeps metadata.
    out = TimeSkew(coefficient=2.0)(events)
    assert out.sensor_size == (64, 48)
    # Through a Compose block too.
    out2 = Compose([TimeSkew(coefficient=2.0), DropEvent(p=0.1)])(events)
    assert out2.sensor_size == (64, 48)


def test_spatial_transform_uses_events_sensor_size():
    events = EventArray(t=[0, 1, 2], x=[0, 10, 63], y=[1, 2, 3], p=[0, 1, 0],
                        metadata={"sensor_size": (64, 48)})
    # No explicit sensor_size: falls back to events metadata (standalone).
    assert list(RandomFlipLR(p=1.0)(events).x) == [63, 53, 0]
    # And inside Compose.
    out = Compose([RandomFlipLR(p=1.0)])(events)
    assert list(out.x) == [63, 53, 0]
    # Explicit sensor_size wins over metadata.
    assert list(RandomFlipLR(sensor_size=(100, 50), p=1.0)(events).x) == [99, 89, 36]


def test_spatial_transform_without_sensor_size_raises():
    events = EventArray(t=[0], x=[1], y=[1], p=[0])  # no metadata
    with pytest.raises(ValueError, match="sensor_size"):
        RandomFlipLR(p=1.0)(events)


def test_compose_survives_emptying_midblock():
    """A drop that empties the stream must not crash a later kernel."""
    events = _make_events(n=100)
    pipeline = Compose([
        DropEvent(p=0.999999),      # almost certainly empties the stream
        RefractoryPeriod(delta=10),  # would call x.max() on empty input
    ])
    out = pipeline(events)          # must not raise
    assert len(out) <= len(events)
