import numpy as np
from evutils.jit import lazy_njit

@lazy_njit
def drop_random_events_jit(t: np.ndarray, x: np.ndarray, y: np.ndarray, p: np.ndarray, drop_rate: float):
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
