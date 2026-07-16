"""Lazy numba compilation helper.

Importing numba costs several hundred milliseconds, which would be paid by
``import evutils.io`` even when no numba-accelerated code path is ever used
(e.g. pure reading through the C parsers). :func:`lazy_njit` defers both the
numba import and the JIT compilation to the first call of the decorated
function.
"""
from __future__ import annotations

import functools
from typing import Callable, TypeVar, Any

F = TypeVar("F", bound=Callable[..., Any])

def lazy_njit(fn: F) -> F:
    """``numba.njit``, but imported and compiled on first call.

    The wrapped function behaves like the ``@nb.njit``-decorated original;
    only the timing of the numba import/compilation differs. The per-call
    overhead after the first call is a single ``is None`` check.
    """
    compiled: Callable[..., Any] | None = None

    @functools.wraps(fn)
    def wrapper(*args: Any, **kwargs) -> Any:
        nonlocal compiled
        if compiled is None:
            import numba as nb
            compiled = nb.njit(fn)
        return compiled(*args, **kwargs)

    return wrapper  # type: ignore[return-value]

def lazy_njit_unwrapped_events(fn: F) -> F:
    """Decorator that unwraps SoA or AoS events into constituent arrays, 
    then calls a lazily compiled numba function. Numba handles the 
    specialization for strided vs contiguous memory.
    """
    compiled: Callable[..., Any] | None = None

    @functools.wraps(fn)
    def wrapper(events: Any, *args: Any, **kwargs) -> Any:
        nonlocal compiled
        if compiled is None:
            import numba as nb
            compiled = nb.njit(fn)
        
        # We import SoaArray here to avoid any top-level circular imports
        from .types import SoaArray
        import numpy as np
        
        if isinstance(events, SoaArray):
            arrays = tuple(getattr(events, f) for f in events._fields)
        elif isinstance(events, np.ndarray) and events.dtype.names is not None:
            arrays = tuple(events[f] for f in events.dtype.names)
        else:
            raise TypeError(f"Unsupported event format: {type(events)}")
            
        return compiled(*arrays, *args, **kwargs)

    return wrapper  # type: ignore[return-value]
