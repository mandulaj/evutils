"""Lazy numba compilation helper.

Importing numba costs several hundred milliseconds, which would be paid by
``import evutils.io`` even when no numba-accelerated code path is ever used
(e.g. pure reading through the C parsers). :func:`lazy_njit` defers both the
numba import and the JIT compilation to the first call of the decorated
function.
"""
from __future__ import annotations

import functools
from typing import Any, Callable, TypeVar

F = TypeVar("F", bound=Callable[..., Any])


def lazy_njit(fn: F) -> F:
    """``numba.njit``, but imported and compiled on first call.

    The wrapped function behaves like the ``@nb.njit``-decorated original;
    only the timing of the numba import/compilation differs. The per-call
    overhead after the first call is a single ``is None`` check.
    """
    compiled: Callable[..., Any] | None = None

    @functools.wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        nonlocal compiled
        if compiled is None:
            import numba as nb
            compiled = nb.njit(fn)
        return compiled(*args, **kwargs)

    return wrapper  # type: ignore[return-value]
