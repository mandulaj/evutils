"""Temporal functional transforms (skew, jitter, normalize)."""
import numpy as np
from evutils.jit import lazy_njit

from ._common import apply_kernel

@lazy_njit
def _time_skew_jit(t, x, y, p, coefficient: float, offset: float):
    """Apply the affine timestamp map ``t' = t * coefficient + offset``."""
    new_t = (t.astype(np.float64) * coefficient + offset).astype(np.int64)
    return new_t, x, y, p

def time_skew(events, coefficient, offset=0.0):
    """Rescale (and shift) all timestamps by a linear map.

    Parameters
    ----------
    events : np.ndarray or EventArray
        Events to skew.
    coefficient : float
        Multiplier applied to every timestamp (e.g. ``2.0`` doubles all gaps).
    offset : float, optional
        Added after multiplication. Default ``0.0``.

    Returns
    -------
    np.ndarray or EventArray
        Events with rewritten timestamps, in their original container type.
    """
    return apply_kernel(events, _time_skew_jit, float(coefficient), float(offset))

@lazy_njit
def _time_jitter_jit(t, x, y, p, std: float, clip_negative: bool,
                     sort_timestamps: bool):
    """Add Gaussian noise to timestamps, optionally clipping and re-sorting."""
    shifts = np.random.normal(0.0, std, len(t))
    new_t = (t.astype(np.float64) + shifts).astype(np.int64)

    if clip_negative:
        keep = new_t >= 0
        new_t, x, y, p = new_t[keep], x[keep], y[keep], p[keep]

    if sort_timestamps:
        order = np.argsort(new_t)
        new_t, x, y, p = new_t[order], x[order], y[order], p[order]

    return new_t, x, y, p

def time_jitter(events, std=1.0, clip_negative=True, sort_timestamps=False):
    """Add Gaussian noise to each timestamp.

    Parameters
    ----------
    events : np.ndarray or EventArray
        Events to jitter.
    std : float, optional
        Standard deviation of the timestamp noise. Default ``1.0``.
    clip_negative : bool, optional
        Drop events whose jittered timestamp is negative. Default ``True``.
    sort_timestamps : bool, optional
        Re-sort events by timestamp after jittering. Default ``False``.

    Returns
    -------
    np.ndarray or EventArray
        Jittered events, in their original container type.
    """
    return apply_kernel(events, _time_jitter_jit, float(std),
                        bool(clip_negative), bool(sort_timestamps))

@lazy_njit
def _normalize_ts_jit(t, x, y, p, start_ts: int):
    """Shift every timestamp so the minimum lands at ``start_ts``."""
    new_t = t - (t.min() - start_ts)
    return new_t, x, y, p

def normalize_ts(events, start_ts=0):
    """Shift timestamps so the earliest event lands at ``start_ts``.

    Pure and stateless: each call normalizes the batch it is handed on its own.
    For chunk-by-chunk streams use ``EventReader(normalize_ts=True)``, which
    latches the offset from the first chunk and applies it across the whole
    stream (so the timeline stays continuous instead of resetting per chunk).

    Parameters
    ----------
    events : np.ndarray or EventArray
        Events to normalize.
    start_ts : int, optional
        Timestamp assigned to the earliest event. Default ``0``.

    Returns
    -------
    np.ndarray or EventArray
        Events with shifted timestamps, in their original container type. The
        input is not modified.

    Examples
    --------
    >>> import numpy as np
    >>> from evutils.transforms.functional import normalize_ts
    >>> events = np.array(
    ...     [(0, 0, 100, 1), (1, 1, 200, 1), (2, 2, 300, 0)],
    ...     dtype=[('x', 'u2'), ('y', 'u2'), ('t', 'i8'), ('p', 'i1')]
    ... )
    >>> normalize_ts(events)['t']
    array([  0, 100, 200])
    """
    return apply_kernel(events, _normalize_ts_jit, int(start_ts))
