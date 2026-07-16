import numpy as np
import pytest
from evutils.utils._checker import EventsChecker
from evutils.types import EventArray

def test_events_checker_accepts_event_array() -> None:
    events = EventArray(t=[1, 2], x=[10, 20], y=[30, 40], p=[1, 0])
    checker = EventsChecker(events)
    assert checker.is_valid()
    
def test_events_checker_repr_performance() -> None:
    events = EventArray(t=[1, 2], x=[10, 20], y=[30, 40], p=[1, 0])
    checker = EventsChecker(events)
    rep = repr(checker)
    assert "EventsChecker(events=2)" == rep
