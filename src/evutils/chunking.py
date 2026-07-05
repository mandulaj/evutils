"""Splitting event streams into chunks.

Slice a continuous event stream into fixed-size windows, either by event
count or by time interval.
"""

import numpy as np



def window_delta_t(events: np.ndarray, delta_t: int = 10_000):
    """Returns a generator that chunks the events array into windows of size delta_t.
    
    Parameters
    ----------
    events : np.ndarray
        Array of events
    delta_t : int, optional
        Size of the window in microseconds, by default 10_000

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

def sliding_window(events: np.ndarray, delta_t: int = 10_000, window_size: int = 20_000, full_window: bool = False):
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




def sort_events(events: np.ndarray):
    """Sorts the events array by timestamp.
    
    Parameters
    ----------
    events : np.ndarray
        Array of events

    """
    return np.sort(events, order="t")

def get_dt_events(events: np.ndarray, dt: int =10_000):
    """Returns the events that are within a time window of dt from the first event's timestamp.

    Parameters
    ----------
    events : np.ndarray
        Array of events
    dt : int, optional
        Time window in microseconds, by default 10_000

    """
    if len(events) == 0:
        return events

    first_ts = events[0]['t'] 
    last_ts = first_ts + dt

    next_index = np.searchsorted(events['t'], last_ts)
    
    return events[:next_index]