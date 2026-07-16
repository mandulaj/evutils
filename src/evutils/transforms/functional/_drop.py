import math

import numpy as np
from evutils.jit import lazy_njit

from ._common import apply_kernel, sample_range

@lazy_njit
def _drop_random_events_jit(t: np.ndarray, x: np.ndarray, y: np.ndarray, p: np.ndarray, drop_rate: float):
    """
    Drops a percentage of events randomly using slicing, compiled via Numba.
    
    Parameters
    ----------
    t, x, y, p : np.ndarray
        Constituent event arrays.
    drop_rate : float
        Percentage of events to drop (0 to 1).
        
    Returns
    -------
    tuple
        (new_t, new_x, new_y, new_p)
    """
    # Using random.rand to generate a boolean mask is fully supported and highly optimized in Numba
    mask = np.random.rand(len(t)) >= drop_rate
    return t[mask], x[mask], y[mask], p[mask]

def drop_random_events(events, drop_rate: float = 0.1):
    """Drops a percentage of events randomly.
    
    Parameters
    ----------
    events : np.ndarray or EventArray
        Array of events to drop from.
    drop_rate : float, optional
        Percentage of events to drop, by default 0.1 (10%).
        
    Returns
    -------
    np.ndarray or EventArray
        Array of events with the specified percentage dropped.
    """
    import math
    if math.isnan(drop_rate) or drop_rate <= 0 or drop_rate >= 1:
        raise ValueError("drop_rate must be between 0 and 1")
        
    from evutils.transforms.compose import unwrap_events, repack_events
    if len(events) == 0:
        return events
        
    t, x, y, p = unwrap_events(events)
    t, x, y, p = _drop_random_events_jit(t, x, y, p, drop_rate)
    return repack_events(events, t, x, y, p)

def drop_event(events, p=0.1):
    """Randomly drop each event independently with probability ``p``.

    The tonic-compatible name for :func:`drop_random_events`. Unlike tonic's
    ``drop_event_numpy`` (which drops exactly ``round(p * n)`` events via
    sampling without replacement) this uses an independent Bernoulli mask per
    event, which is what keeps the kernel JIT-friendly. The expected number of
    dropped events is the same; the exact count is binomially distributed.

    Parameters
    ----------
    events : np.ndarray or EventArray
        Events to drop from.
    p : float or tuple of float, optional
        Drop probability in ``[0, 1)``. A ``(lo, hi)`` tuple is sampled uniformly
        per call. Defaults to ``0.1``.

    Returns
    -------
    np.ndarray or EventArray
        Events that survived the drop, in their original container type.
    """
    p = sample_range(p)
    if math.isnan(p) or p < 0 or p >= 1:
        raise ValueError("p must be in [0, 1)")
    if p == 0:
        return events
    return apply_kernel(events, _drop_random_events_jit, p)

@lazy_njit
def _drop_by_time_jit(t, x, y, p, duration_ratio: float):
    """Drop a single contiguous time window covering ``duration_ratio`` of the span."""
    t_end = t.max()
    drop_duration = t_end * duration_ratio
    hi = t_end - drop_duration
    # np.random.uniform requires low < high; clamp the degenerate case.
    drop_start = np.random.uniform(0.0, hi) if hi > 0.0 else 0.0
    keep = (t < drop_start) | (t > drop_start + drop_duration)
    return t[keep], x[keep], y[keep], p[keep]

def drop_by_time(events, duration_ratio=0.2):
    """Drop every event inside one randomly-placed time window.

    The window length is ``duration_ratio`` of the recording span (``[0, t.max()]``,
    following tonic), positioned uniformly at random within it.

    Parameters
    ----------
    events : np.ndarray or EventArray
        Events to drop from.
    duration_ratio : float or tuple of float, optional
        Window length as a fraction of the span, in ``[0, 1)``. A ``(lo, hi)``
        tuple is sampled uniformly per call. Defaults to ``0.2``.

    Returns
    -------
    np.ndarray or EventArray
        Events outside the dropped window, in their original container type.
    """
    ratio = sample_range(duration_ratio)
    if math.isnan(ratio) or ratio < 0 or ratio >= 1:
        raise ValueError("duration_ratio must be in [0, 1)")
    if ratio == 0:
        return events
    return apply_kernel(events, _drop_by_time_jit, ratio)
