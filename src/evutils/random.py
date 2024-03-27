
from typing import Generator
from .types import Events
import numpy as np 

def random_events(n_events: int, width: int = 1280, height: int = 720, start_ts: int = 0, end_ts: int = 10_000_000) -> np.ndarray:
    '''Generates n_events random events with x and y coordinates in the range [0, width) and [0, height) respectively'''
    
    events = np.empty(n_events, dtype=Events)
    events["x"] = np.random.randint(0, width, n_events)
    events["y"] = np.random.randint(0, height, n_events)
    events["p"] = np.random.randint(0, 2, n_events)
    events["t"] = np.random.randint(start_ts, end_ts, n_events)

    # Sort the timestamps
    events["t"].sort()

    return events


def random_events_generator(n_events: int, width: int = 1280, height: int = 720, start_ts: int = 0, end_ts: int = 10_000_000, chunk_size=10000) -> Generator[np.ndarray, None, None]:
    '''Generates n_events random events with x and y coordinates in the range [0, width) and [0, height) respectively'''
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