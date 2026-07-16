"""Utility functions for processing event arrays."""

import numpy as np


def normalize_ts(events: np.ndarray, start_ts: int = 0) -> np.ndarray:
    """Normalizes the timestamps of events to start from zero.

    Parameters
    ----------
    events : np.ndarray
        Array of events with a 't' field representing timestamps.
    start_ts : int, optional
        The timestamp to normalize from, by default 0.

    Returns
    -------
    np.ndarray
        Array of events with normalized timestamps.

    Examples
    --------
    >>> import numpy as np
    >>> from evutils.processing import normalize_ts
    >>> events = np.array(
    ...     [(0, 0, 100, 1), (1, 1, 200, 1), (2, 2, 300, 0)],
    ...     dtype=[('x', 'u2'), ('y', 'u2'), ('t', 'i8'), ('p', 'i1')]
    ... )
    >>> norm_events = normalize_ts(events.copy())
    >>> norm_events['t']
    array([  0, 100, 200])

    """
    if len(events) == 0:
        return events

    if not (hasattr(events, 'dtype') and events.dtype.names and 't' in events.dtype.names):
        if not hasattr(events, 't'):
            raise ValueError("events must be a structured array or object with a 't' field")

    if hasattr(events, 'flags') and not events.flags.writeable:
        events = events.copy()

    min_ts = events['t'].min()
    events['t'] -= min_ts - start_ts

    return events