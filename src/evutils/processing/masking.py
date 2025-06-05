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
    """

    # Check if mask is a 2D array
    if mask.ndim != 2:
        raise ValueError("Mask must be a 2D array")
    
    # Check if max x and y in events are within the mask dimensions
    if events['x'].max() < 0 or events['y'].max() < 0:
        raise ValueError("Events x and y coordinates must be non-negative")
    if events['x'].max() >= mask.shape[1] or events['y'].max() >= mask.shape[0]:
        raise ValueError("Events x and y coordinates must be within the mask dimensions")

    x = events['x']
    y = events['y']
    valid_events = mask[y, x] > 0
    
    return events[valid_events]