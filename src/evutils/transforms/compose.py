import numpy as np
from typing import Iterable, Callable, Union
from evutils.types import SoaArray

def unwrap_events(events: Union[np.ndarray, SoaArray]):
    """Unwraps events into a tuple of (t, x, y, p) arrays."""
    if isinstance(events, SoaArray):
        return events.t, events.x, events.y, events.p
    elif isinstance(events, np.ndarray) and events.dtype.names is not None:
        return events['t'], events['x'], events['y'], events['p']
    else:
        raise TypeError(f"Unsupported event format: {type(events)}")

def repack_events(original_events: Union[np.ndarray, SoaArray], t: np.ndarray, x: np.ndarray, y: np.ndarray, p: np.ndarray) -> Union[np.ndarray, SoaArray]:
    """Repacks raw arrays back into the user's original format."""
    if isinstance(original_events, SoaArray):
        return original_events.__class__(t=t, x=x, y=y, p=p)
    elif isinstance(original_events, np.ndarray) and original_events.dtype.names is not None:
        new_events = np.empty(len(t), dtype=original_events.dtype)
        new_events['t'] = t
        new_events['x'] = x
        new_events['y'] = y
        new_events['p'] = p
        return new_events
    else:
        raise TypeError(f"Unsupported event format: {type(original_events)}")

class Compose:
    """Composes several transforms together.
    
    Groups contiguous blocks of evutils `Transform` objects to execute them purely in 
    C-space via Numba JIT without intermediate unpacking/repacking overhead.
    Freely accepts standard Callables (e.g. PyTorch/Tonic transforms).
    """

    def __init__(self, transforms: Iterable[Callable]):
        self.transforms = list(transforms)
        self._execution_plan = []
        
        # Pre-compute the JIT blocks
        current_jit_block = []
        for t in self.transforms:
            if hasattr(t, "_forward_jit"):
                current_jit_block.append(t)
            else:
                if current_jit_block:
                    self._execution_plan.append(("jit", current_jit_block))
                    current_jit_block = []
                self._execution_plan.append(("standard", t))
                
        if current_jit_block:
             self._execution_plan.append(("jit", current_jit_block))

    def __call__(self, events, target=None):
        import inspect
        for step_type, block in self._execution_plan:
            if len(events) == 0:
                break
                
            if step_type == "jit":
                t, x, y, p = unwrap_events(events)
                for transform in block:
                    t, x, y, p = transform._forward_jit(t, x, y, p)
                    if target is not None:
                        target = transform._transform_target(target)
                events = repack_events(events, t, x, y, p)
            else:
                if target is not None:
                    # Check if standard block accepts a target
                    sig = inspect.signature(block)
                    if len(sig.parameters) > 1:
                        res = block(events, target)
                        if isinstance(res, tuple) and len(res) == 2:
                            events, target = res
                        else:
                            events = res
                    else:
                        events = block(events)
                else:
                    events = block(events)
                
        if target is not None:
            return events, target
        return events

    def __repr__(self):
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += f"    {t}"
        format_string += "\n)"
        return format_string
