


from ._histogram import histogram
from ..chunking import window_delta_t


import numpy as np


def voxel_histogram(events: np.ndarray, width: int = 1280, height: int = 720, n_bins: int = 10, dt: int = 10_000, dtype=np.uint8) -> np.ndarray:
    """
    Generate a voxel grid from the events.

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
    dtype : type, optional
        Data type of the output voxel grid, by default np.uint8.

    Returns
    -------
    out : np.ndarray
        A numpy array with the voxel grid (height, width, depth).
    """
    
    buffer = np.zeros((n_bins, height, width, 3), dtype=dtype)


    if len(events) <= 2:
        return buffer

    assert events[-1]['t'] - events[0]['t'] <= dt # Ensure that the events are within the specified time delta

    bin_dt = dt // n_bins  # Time per bin in microseconds

    for i, e in enumerate(window_delta_t(events, delta_t=bin_dt)):
        hist = histogram(e, width=width, height=height, fill=False, dtype=dtype)

        # Only keep the r and b channels
        buffer[i] = hist



    return buffer
