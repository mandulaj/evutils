"""Spatial functional transforms (flips, jitter)."""
import math

import numpy as np
from evutils.jit import lazy_njit

from ._common import apply_kernel

@lazy_njit
def _flip_lr_jit(t, x, y, p, width: int):
    """Mirror x about the sensor width: ``x' = width - 1 - x``."""
    # Promote through int64 so the uint16 subtraction can't wrap around.
    new_x = (np.int64(width) - 1 - x.astype(np.int64)).astype(np.uint16)
    return t, new_x, y, p

def flip_lr(events, sensor_size):
    """Flip events horizontally: ``x' = width - 1 - x``.

    Parameters
    ----------
    events : np.ndarray or EventArray
        Events to flip.
    sensor_size : tuple
        ``(W, H, P)`` sensor size; only the width ``W`` is used.

    Returns
    -------
    np.ndarray or EventArray
        Horizontally flipped events, in their original container type.
    """
    return apply_kernel(events, _flip_lr_jit, int(sensor_size[0]))

@lazy_njit
def _spatial_jitter_jit(t, x, y, p, width: int, height: int,
                        var_x: float, var_y: float, sigma_xy: float,
                        clip_outliers: bool):
    """Add correlated Gaussian noise to ``x``/``y``.

    Numba has no ``multivariate_normal``, so we sample two independent standard
    normals and correlate them with the Cholesky factor of
    ``[[var_x, sigma_xy], [sigma_xy, var_y]]``.
    """
    n = len(x)
    l11 = math.sqrt(var_x)
    l21 = sigma_xy / l11 if l11 > 0.0 else 0.0
    under = var_y - l21 * l21
    l22 = math.sqrt(under) if under > 0.0 else 0.0

    z1 = np.random.normal(0.0, 1.0, n)
    z2 = np.random.normal(0.0, 1.0, n)
    new_x = x.astype(np.float64) + l11 * z1
    new_y = y.astype(np.float64) + (l21 * z1 + l22 * z2)

    if clip_outliers:
        keep = (new_x >= 0) & (new_x < width) & (new_y >= 0) & (new_y < height)
        t, p = t[keep], p[keep]
        new_x, new_y = new_x[keep], new_y[keep]

    return t, new_x.astype(np.uint16), new_y.astype(np.uint16), p

def spatial_jitter(events, sensor_size, var_x=1.0, var_y=1.0, sigma_xy=0.0,
                   clip_outliers=False):
    """Jitter event coordinates with a 2D Gaussian.

    Parameters
    ----------
    events : np.ndarray or EventArray
        Events to jitter.
    sensor_size : tuple
        ``(W, H, P)`` sensor size, used for clipping.
    var_x, var_y : float, optional
        Variances of the jitter in x and y. Default ``1.0``.
    sigma_xy : float, optional
        Off-diagonal covariance (diagonal skew). Default ``0.0``.
    clip_outliers : bool, optional
        Drop events jittered outside the sensor instead of casting them back
        (which would wrap the unsigned coordinate). Default ``False``.

    Returns
    -------
    np.ndarray or EventArray
        Jittered events, in their original container type.
    """
    return apply_kernel(events, _spatial_jitter_jit, int(sensor_size[0]),
                        int(sensor_size[1]), float(var_x), float(var_y),
                        float(sigma_xy), bool(clip_outliers))
