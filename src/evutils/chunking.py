"""Splitting event streams into chunks.

Slice a continuous event stream into fixed-size windows, either by event
count or by time interval.
"""

import numpy as np
from typing import Iterator



def window_delta_t(events: np.ndarray, delta_t: int = 10_000) -> Iterator[np.ndarray]:
    """Returns a generator that chunks the events array into windows of size delta_t.
    
    Parameters
    ----------
    events : np.ndarray
        Array of events
    delta_t : int, optional
        Size of the window in microseconds, by default 10_000

    Examples
    --------
    >>> from evutils.random import random_events
    >>> from evutils.chunking import window_delta_t
    >>> events = random_events(1000, start_ts=0, end_ts=30_000)
    >>> chunks = list(window_delta_t(events, delta_t=10_000))
    >>> len(chunks) > 0
    True
    """
    if len(events) == 0:
        return

    index_start = 0
    
    ts = events["t"]
    current_ts = ts[0]

    while index_start < len(events):
        next_index = np.searchsorted(ts[index_start:], current_ts + delta_t)

        window = events[index_start:index_start + next_index]
        yield window

        current_ts += delta_t
        index_start += next_index

def sliding_window(events: np.ndarray, delta_t: int = 10_000, window_size: int = 20_000, full_window: bool = False) -> Iterator[np.ndarray]:
    """Returns a generator that chunks the events array into windows of size delta_t.
    
    Parameters
    ----------
    events : np.ndarray
        Array of events
    delta_t : int, optional
        Time delta between frames in microseconds, by default 10_000
    window_size : int, optional
        Size of the window in microseconds, by default 20_000
        can overlap with the next frame
    full_window : bool, optional
        If True, the last window will be full, by default False
        If False, the last window will be the remaining events

    Examples
    --------
    >>> from evutils.random import random_events
    >>> from evutils.chunking import sliding_window
    >>> events = random_events(1000, start_ts=0, end_ts=50_000)
    >>> chunks = list(sliding_window(events, delta_t=10_000, window_size=20_000))
    >>> len(chunks) > 0
    True
    """
    if len(events) == 0:
        return

    index_start = 0
    
    ts = events["t"]
    current_ts = ts[0]

    while index_start < len(events):


        next_frame_index = np.searchsorted(ts[index_start:], current_ts + delta_t)
        next_window_index = np.searchsorted(ts[index_start:], current_ts + window_size)
        
       

        window = events[index_start:index_start + next_window_index]
        yield window


        # Exit if the next window index is  not full
        if full_window and index_start + next_window_index >= len(events):
            break

        current_ts += delta_t
        index_start += next_frame_index




def sort_events(events: np.ndarray) -> np.ndarray:
    """Sorts the events array by timestamp.
    
    Parameters
    ----------
    events : np.ndarray
        Array of events

    Examples
    --------
    >>> import numpy as np
    >>> from evutils.chunking import sort_events
    >>> from evutils.random import random_events
    >>> events = random_events(10)
    >>> events["t"] = np.arange(10, 0, -1)
    >>> sorted_events = sort_events(events)
    >>> int(sorted_events["t"][0])
    1
    """
    return np.sort(events, order="t")

def get_dt_events(events: np.ndarray, dt: int =10_000) -> np.ndarray:
    """Returns the events that are within a time window of dt from the first event's timestamp.

    Parameters
    ----------
    events : np.ndarray
        Array of events
    dt : int, optional
        Time window in microseconds, by default 10_000

    Examples
    --------
    >>> import numpy as np
    >>> from evutils.random import random_events
    >>> from evutils.chunking import get_dt_events
    >>> events = random_events(100, start_ts=0, end_ts=50_000)
    >>> sub_events = get_dt_events(events, dt=10_000)
    >>> bool((sub_events["t"] <= events["t"][0] + 10_000).all())
    True
    """
    if len(events) == 0:
        return events

    first_ts = events[0]['t'] 
    last_ts = first_ts + dt

    next_index = np.searchsorted(events['t'], last_ts)
    
    return events[:next_index]