"""Shared helpers for the functional transforms.

Every functional follows the same shape: validate arguments, then unwrap the
events into the constituent ``(t, x, y, p)`` arrays, run a Numba-compiled
kernel, and repack the result into the caller's original container. The
:func:`apply_kernel` helper centralises the unwrap/kernel/repack dance so each
functional only has to express its argument handling and pick a kernel.
"""
from __future__ import annotations

from typing import Callable

import numpy as np

def apply_kernel(events: "EventArray", kernel: Callable[..., "EventArray"], *args: object) -> "EventArray":
    """Unwrap ``events``, run ``kernel(t, x, y, p, *args)``, repack the result.

    Empty inputs are returned untouched so kernels never see zero-length arrays
    (several derive a sensor extent from ``x.max()`` / ``y.max()``).
    """
    from evutils.transforms.compose import repack_events, unwrap_events

    if len(events) == 0:
        return events

    t, x, y, p = unwrap_events(events)
    t, x, y, p = kernel(t, x, y, p, *args)
    return repack_events(events, t, x, y, p)

def sample_range(value: "tuple[float, float] | list[float] | float") -> float:
    """Return ``value``, or a uniform sample in ``[lo, hi)`` if it is a 2-tuple.

    Mirrors the range-sampling convention used across the tonic transforms,
    where a scalar is used verbatim and a ``(lo, hi)`` pair is sampled per call.
    """
    if isinstance(value, (tuple, list)):
        lo, hi = value
        return (hi - lo) * np.random.random_sample() + lo
    return float(value)
