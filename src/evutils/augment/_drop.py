"""Module for augmenting events by dropping random events."""

import numpy as np



def drop_random_events(events: np.ndarray, drop_rate: float = 0.1) -> np.ndarray:
    """Drops a percentage of events randomly using slicing.
    
    Parameters
    ----------
    events : np.ndarray
        Array of events to drop from.
    drop_rate : float, optional
        Percentage of events to drop, by default 0.1 (10%).
    
    Returns
    -------
    np.ndarray
        Array of events with the specified percentage dropped.

    """
    import math
    if math.isnan(drop_rate) or drop_rate <= 0 or drop_rate >= 1:
        raise ValueError("drop_rate must be between 0 and 1")
    
    n_events = len(events)
    indices = np.arange(n_events)
    np.random.shuffle(indices)
    
    drop_count = int(n_events * drop_rate)
    keep_indices = indices[drop_count:]
    
    return events[keep_indices]