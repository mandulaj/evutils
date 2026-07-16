"""Generation and perturbation of synthetic events.

Create random event arrays for testing and benchmarking
(``random_events``, ``random_events_generator``) and add random timestamp
jitter to existing events (``event_jitter``, ``event_jitter_n``).
"""

from typing import Generator
from .types import Event_dtype
import numpy as np 

def random_events(n_events: int, width: int = 1280, height: int = 720, start_ts: int = 0, end_ts: int = 10_000_000) -> np.ndarray:
    """Generates n_events random events with x and y coordinates in the range [0, width) and [0, height) respectively.

    Examples
    --------
    >>> import numpy as np
    >>> from evutils.random import random_events
    >>> events = random_events(10, width=640, height=480, start_ts=0, end_ts=1000)
    >>> len(events)
    10
    >>> bool(events["x"].max() < 640)
    True
    """
    events = np.empty(n_events, dtype=Event_dtype)
    events["x"] = np.random.randint(0, width, n_events)
    events["y"] = np.random.randint(0, height, n_events)
    events["p"] = np.random.randint(0, 2, n_events)
    events["t"] = np.random.randint(start_ts, end_ts, n_events)

    # Sort the timestamps
    events["t"].sort()

    return events

def random_events_generator(n_events: int, width: int = 1280, height: int = 720, start_ts: int = 0, end_ts: int = 10_000_000, chunk_size: int = 10000) -> Generator[np.ndarray, None, None]:
    """Generates n_events random events with x and y coordinates in the range [0, width) and [0, height) respectively.

    Examples
    --------
    >>> from evutils.random import random_events_generator
    >>> gen = random_events_generator(25000, chunk_size=10000)
    >>> for chunk in gen:
    ...     print(len(chunk))
    10000
    10000
    5000
    """
    if n_events == 0:
        return
        
    n_chunks = int(np.ceil(n_events / chunk_size))

    chunk_ts_len = (end_ts - start_ts) // n_chunks
    if chunk_ts_len == 0:
        chunk_ts_len = 1

    chunk_end_ts = start_ts + chunk_ts_len

    for chunk in range(n_chunks):
        if chunk >= n_chunks - 1:
            chunk_end_ts = end_ts
            chunk_size = n_events - (chunk * chunk_size)
        

        events = random_events(chunk_size, width, height, start_ts, chunk_end_ts)

        start_ts += chunk_ts_len
        chunk_end_ts += chunk_ts_len

        yield events

def event_jitter_n(events: np.ndarray, mean: float = 0.0, std: float = 1.0, sort: bool = True, in_place: bool = False) -> np.ndarray:
    """Adds a random jitter to the timestamps of the events.

    Examples
    --------
    >>> import numpy as np
    >>> from evutils.random import random_events, event_jitter_n
    >>> events = random_events(5)
    >>> jittered = event_jitter_n(events.copy(), mean=0.0, std=5.0)
    >>> len(jittered) == len(events)
    True
    """
    if not in_place:
        events = events.copy()
    events["t"] += np.round(np.random.normal(mean, std, len(events))).astype(np.int64)

    if sort:
        events = np.sort(events, order="t")

    return events

def event_jitter(events: np.ndarray, jitter: int = 1, sort: bool = True, in_place: bool = False) -> np.ndarray:
    """Adds a random jitter to the timestamps of the events.

    Examples
    --------
    >>> import numpy as np
    >>> from evutils.random import random_events, event_jitter
    >>> events = random_events(10)
    >>> jittered = event_jitter(events.copy(), jitter=5, sort=True)
    >>> len(jittered) == len(events)
    True
    """
    jitter = int(jitter)
    
    if not in_place:
        events = events.copy()
    events["t"] += np.random.randint(-jitter, jitter, len(events))

    if sort:
        events = np.sort(events, order="t")

    return events

