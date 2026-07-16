import numpy as np
from evutils.jit import lazy_njit

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
