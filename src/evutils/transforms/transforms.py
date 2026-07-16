import math
from typing import Union
import numpy as np
from evutils.types import SoaArray

class Transform:
    """Base class for all evutils transforms.
    
    Transforms should implement the `_forward_jit` method to allow zero-overhead 
    composition inside a `Compose` pipeline.
    """
    def __call__(self, events: Union[np.ndarray, SoaArray], target=None):
        """Applies the transform in a standalone manner."""
        from evutils.transforms.compose import unwrap_events, repack_events
        if len(events) == 0:
            if target is not None:
                return events, target
            return events
            
        t, x, y, p = unwrap_events(events)
        t, x, y, p = self._forward_jit(t, x, y, p)
        events = repack_events(events, t, x, y, p)
        
        if target is not None:
            target = self._transform_target(target)
            return events, target
        return events
        
    def _forward_jit(self, t: np.ndarray, x: np.ndarray, y: np.ndarray, p: np.ndarray):
        """The pure array-math forward pass, ideally JIT-compiled."""
        raise NotImplementedError("Transforms must implement _forward_jit")

    def _transform_target(self, target):
        """Pure Python transformation of the target (e.g. bounding boxes).
        Defaults to doing nothing.
        """
        return target

class DropRandomEvents(Transform):
    """Drops a percentage of events randomly.
    
    Parameters
    ----------
    drop_rate : float, optional
        Percentage of events to drop, by default 0.1 (10%).
    """
    def __init__(self, drop_rate: float = 0.1):
        if math.isnan(drop_rate) or drop_rate <= 0 or drop_rate >= 1:
            raise ValueError("drop_rate must be between 0 and 1")
        self.drop_rate = drop_rate

    def _forward_jit(self, t, x, y, p):
        from evutils.transforms.functional import _drop_random_events_jit
        return _drop_random_events_jit(t, x, y, p, self.drop_rate)
        
    def __repr__(self):
        return f"{self.__class__.__name__}(drop_rate={self.drop_rate})"
