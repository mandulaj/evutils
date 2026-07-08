"""Module for applying spatial masks to event arrays."""

import numpy as np


def mask_events(events: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Masks events based on a given mask.

    Parameters
    ----------
    events : np.ndarray
        Array of events to be masked.
    mask : np.ndarray
        A 2D mask array where the events will be checked against. 
        The mask should have the same shape as the event frame size.
        
    Returns
    -------
    np.ndarray
        Array of events that fall within the valid regions of the mask.

    Examples
    --------
    >>> import numpy as np
    >>> from evutils.processing import mask_events
    >>> events = np.array(
    ...     [(0, 0, 100, 1), (1, 1, 200, 1), (2, 2, 300, 0)],
    ...     dtype=[('x', 'u2'), ('y', 'u2'), ('t', 'i8'), ('p', 'i1')]
    ... )
    >>> mask = np.array([
    ...     [1, 0, 0],
    ...     [0, 0, 0],
    ...     [0, 0, 1]
    ... ])
    >>> masked_events = mask_events(events, mask)
    >>> masked_events[['x', 'y']].tolist()
    [(0, 0), (2, 2)]

    """
    # Check if mask is a 2D array
    if mask.ndim != 2:
        raise ValueError("Mask must be a 2D array")

    if len(events) == 0:
        return events
    
    # Check if max x and y in events are within the mask dimensions
    if events['x'].max() < 0 or events['y'].max() < 0:
        raise ValueError("Events x and y coordinates must be non-negative")
    if events['x'].max() >= mask.shape[1] or events['y'].max() >= mask.shape[0]:
        raise ValueError("Events x and y coordinates must be within the mask dimensions")

    x = events['x']
    y = events['y']
    valid_events = mask[y, x] > 0
    
    return events[valid_events]