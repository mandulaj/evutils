

import numpy as np



def window_delta_t(events: np.ndarray, delta_t: int = 10_000):
    '''
    Returns a generator that chunks the events array into windows of size delta_t
    
    Parameters
    ----------
    events : np.ndarray
        Array of events
    delta_t : int, optional
        Size of the window in microseconds, by default 10_000'''


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
        index_start = next_index



def sort_events(events: np.ndarray):
    '''Sorts the events array by timestamp
    
    Parameters
    ----------
    events : np.ndarray
        Array of events

    '''
    return np.sort(events, order="t")