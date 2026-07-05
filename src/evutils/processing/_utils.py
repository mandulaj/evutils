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

    """
    if len(events) == 0:
        return events

    min_ts = events['t'].min()
    events['t'] -= min_ts - start_ts

    return events