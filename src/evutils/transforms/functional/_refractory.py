"""Refractory-period functional transform."""
import numpy as np
from evutils.jit import lazy_njit

from ._common import apply_kernel

@lazy_njit
def _refractory_period_jit(t, x, y, p, delta: int):
    """Discard events that fire within ``delta`` of the previous event at the same pixel.

    An event survives when ``t - t_last > delta`` for its pixel. The per-pixel
    clock is updated on *every* event (including dropped ones), matching tonic's
    ``refractory_period_numpy``. This loop is exactly the kind of scalar-heavy
    code Numba turns into tight machine code.
    """
    n = len(t)
    width = int(x.max()) + 1
    height = int(y.max()) + 1
    # Init to -delta so the first event at each pixel is always kept.
    last = np.zeros((width, height), dtype=np.int64) - delta
    keep = np.zeros(n, dtype=np.bool_)

    for i in range(n):
        xi = x[i]
        yi = y[i]
        if t[i] - last[xi, yi] > delta:
            keep[i] = True
        last[xi, yi] = t[i]

    return t[keep], x[keep], y[keep], p[keep]

def refractory_period(events, delta):
    """Enforce a per-pixel refractory period.

    Parameters
    ----------
    events : np.ndarray or EventArray
        Events to filter. Must be sorted by timestamp.
    delta : int
        Refractory period in the same time unit as the timestamps. Events at a
        pixel that fire within ``delta`` of that pixel's previous event are
        dropped.

    Returns
    -------
    np.ndarray or EventArray
        Filtered events, in their original container type.
    """
    return apply_kernel(events, _refractory_period_jit, int(delta))
