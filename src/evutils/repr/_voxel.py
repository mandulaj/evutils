
"""Module for generating voxel grid representations from events."""

from ._histogram import histogram
from ..chunking import window_delta_t


import numpy as np


from typing import Any
from ..types import EventArray

def voxel_histogram(events: 'np.ndarray | EventArray', width: int = 1280, height: int = 720, n_bins: int = 10, dt: int = 10_000, dtype: Any = np.uint8) -> np.ndarray:
    """Generate a voxel grid from the events.

    Parameters
    ----------
    events : np.ndarray
        Array of events in the :class:`~evutils.types.Events` format.
    width : int, optional
        Width of the voxel grid, by default 1280.
    height : int, optional
        Height of the voxel grid, by default 720.
    n_bins : int, optional
        Number of depth bins (time slices) in the voxel grid, by default 10.
    dt : int, optional
        Time delta in microseconds for events buffer by default 10_000 (10 ms).
    dtype : np.dtype, optional
        Data type of the output voxel grid, by default np.uint8.

    Returns
    -------
    np.ndarray
        A numpy array with the voxel grid (n_bins, height, width, 3).

    Examples
    --------
    >>> import numpy as np
    >>> from evutils.repr import voxel_histogram
    >>> events = np.array([(10, 20, 100, 1), (15, 25, 200, 0), (20, 30, 10000, 1)],
    ...                   dtype=[('x', '<u2'), ('y', '<u2'), ('t', '<i8'), ('p', 'i1')])
    >>> grid = voxel_histogram(events, width=100, height=100, n_bins=10, dt=10000)
    >>> grid.shape
    (10, 100, 100, 3)
    """
    buffer = np.zeros((n_bins, height, width, 3), dtype=dtype)


    if len(events) <= 2:
        return buffer

    if events['t'][-1] - events['t'][0] > dt:
        raise ValueError(f"Events span a duration greater than dt ({dt}).")

    bin_dt = dt // n_bins  # Time per bin in microseconds

    for i, e in enumerate(window_delta_t(events, delta_t=bin_dt)):
        if i >= n_bins:
            break
        hist = histogram(e, width=width, height=height, fill=False, dtype=dtype)

        # Only keep the r and b channels
        buffer[i] = hist



    return buffer
