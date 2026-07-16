"""Splitting event streams into chunks.

Slice a continuous event stream into fixed-size windows, either by event
count or by time interval.
"""

import numpy as np

import time
import queue
import threading
from typing import Iterator, Any
from evutils.io.buffer import EventAccumulator




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
    if delta_t <= 0:
        raise ValueError("delta_t must be positive")
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
    if delta_t <= 0 or window_size <= 0:
        raise ValueError("delta_t and window_size must be positive")
    if len(events) == 0:
        return

    index_start = 0
    
    ts = events["t"]
    current_ts = ts[0]

    while index_start < len(events):


        next_frame_index = np.searchsorted(ts[index_start:], current_ts + delta_t)
        next_window_index = np.searchsorted(ts[index_start:], current_ts + window_size)
        
       

        # Exit if the next window index is not full
        if full_window and index_start + next_window_index >= len(events):
            break

        window = events[index_start:index_start + next_window_index]
        yield window

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
def stream_delta_t(raw_stream: Iterator[Any], delta_t: int) -> Iterator[Any]:
    """A pipeline generator that turns a raw stream into perfect delta_t chunks.
    This maintains a small internal buffer for events that cross boundaries.
    """
    if delta_t <= 0:
        raise ValueError("delta_t must be positive")
    # Start small: the accumulator grows geometrically on demand, so a large
    # up-front capacity only page-faults ~GBs of memory for nothing.
    acc = EventAccumulator(capacity=1_000_000)
    current_ts = None
    has_triggers = False  # set from the stream's items: tuple => (events, triggers)

    for incoming in raw_stream:
        # Handle unpacking depending on if the stream yields triggers or not
        if isinstance(incoming, tuple):
            has_triggers = True
            ev, tr = incoming
            acc.append(ev, tr)
        else:
            acc.append(incoming, None)

        if len(acc) == 0 and (not has_triggers or (acc._tr.size - acc._tr_start) == 0):
            continue

        # Initialize our absolute time anchor from the very first event or trigger
        if current_ts is None:
            if len(acc) > 0:
                current_ts = int(acc.t_window()[0])
            elif has_triggers and (acc._tr.size - acc._tr_start) > 0:
                current_ts = int(acc.t_window_tr()[0])
            else:
                continue

        # Yield as many full windows as we have accumulated
        while True:
            end_ts = current_ts + delta_t
            t = acc.t_window()
            
            max_ts = -1
            if len(t) > 0:
                max_ts = t[-1]
            if has_triggers:
                tr_t = acc.t_window_tr()
                if len(tr_t) > 0 and tr_t[-1] > max_ts:
                    max_ts = tr_t[-1]

            if max_ts < end_ts:
                # Not enough data for a full window yet; fetch more chunks
                break

            # Find boundary
            idx = int(np.searchsorted(t, end_ts, side='left')) if len(t) > 0 else 0

            # Slice and yield. Mirror the input shape: only a trigger-carrying
            # stream ((events, triggers) tuples) yields tuples back.
            if has_triggers:
                tr_t = acc.t_window_tr()
                tr_idx = int(np.searchsorted(tr_t, end_ts, side='left')) if len(tr_t) > 0 else 0
                chunk_ev, chunk_tr = acc.slice_copy(idx, tr_idx)
                yield chunk_ev, chunk_tr
            else:
                chunk_ev, _ = acc.slice_copy(idx, 0)
                yield chunk_ev

            current_ts += delta_t

    # Stream finished! Yield whatever is leftover in the buffer
    if len(acc) > 0:
        if has_triggers:
            yield acc.slice_copy(len(acc), acc._tr.size - acc._tr_start)
        else:
            yield acc.slice_copy(len(acc), 0)[0]


def stream_n_events(raw_stream: Iterator[Any], n_events: int) -> Iterator[Any]:
    """Pipeline generator: chunks stream by event count."""
    if n_events <= 0:
        raise ValueError("n_events must be positive")
    acc = EventAccumulator(capacity=max(1_000_000, n_events * 2))
    has_triggers = False  # set from the stream's items: tuple => (events, triggers)
    for incoming in raw_stream:
        if isinstance(incoming, tuple):
            has_triggers = True
            acc.append(incoming[0], incoming[1])
        else:
            acc.append(incoming, None)

        while len(acc) >= n_events:
            if has_triggers:
                if len(acc) == n_events:
                    tr_idx = acc._tr.size - acc._tr_start
                else:
                    tr_idx = int(np.searchsorted(acc.t_window_tr(), acc.t_window()[n_events], side='left'))
                yield acc.slice_copy(n_events, tr_idx)
            else:
                yield acc.slice_copy(n_events, 0)[0]

    if len(acc) > 0:
        if has_triggers:
            yield acc.slice_copy(len(acc), acc._tr.size - acc._tr_start)
        else:
            yield acc.slice_copy(len(acc), 0)[0]

def stream_skip_to_time(stream: Iterator[Any], start_ts: int) -> Iterator[Any]:
    """Pipeline generator: drops events until start_ts is reached."""
    skipping = True
    for incoming in stream:
        ev = incoming[0] if isinstance(incoming, tuple) else incoming
        if skipping:
            if len(ev) == 0 or ev.t[-1] < start_ts:
                continue  # Drop whole chunk
            
            # Found the boundary! Slice the chunk and stop skipping
            idx = int(np.searchsorted(ev.t, start_ts))
            skipping = False
            
            if isinstance(incoming, tuple):
                tr_idx = int(np.searchsorted(incoming[1].t, start_ts))
                yield incoming[0][idx:], incoming[1][tr_idx:]
            else:
                yield incoming[idx:]
        else:
            yield incoming

def stream_async(stream: Iterator[Any], maxsize: int = 5) -> Iterator[Any]:
    """Pipeline generator: runs upstream decoding in a background thread."""
    q: queue.Queue[Any] = queue.Queue(maxsize=maxsize)
    
    def worker() -> None:
        try:
            for item in stream:
                # IMPORTANT: C-parsers reuse internal buffers! We MUST copy the chunk
                # before placing it in the queue to prevent the next read_chunk() 
                # from overwriting the memory of the chunk we just yielded!
                if isinstance(item, tuple):
                    ev = item[0].copy() if item[0] is not None else None
                    tr = item[1].copy() if item[1] is not None else None
                    q.put((ev, tr))
                else:
                    q.put(item.copy())
        except Exception as e:
            q.put(e)
        finally:
            q.put(None)  # Sentinel
        
    t = threading.Thread(target=worker, daemon=True)
    t.start()
    
    while True:
        item = q.get()
        if item is None:
            break
        if isinstance(item, Exception):
            raise item
        yield item

def stream_paced_playback(stream: Iterator[Any], playback_speed: float = 1.0) -> Iterator[Any]:
    """Pipeline generator: spaces out yielding chunks to match wall-clock real-time."""
    start_wall = None
    start_ts = None
    
    for incoming in stream:
        ev = incoming[0] if isinstance(incoming, tuple) else incoming
        if len(ev) == 0:
            yield incoming
            continue
            
        if start_ts is None:
            start_ts = ev.t[0]
            start_wall = time.perf_counter()
            
        # How far into the stream is this chunk's end?
        stream_elapsed_us = ev.t[-1] - start_ts
        expected_wall_elapsed = (stream_elapsed_us / 1_000_000) / playback_speed
        
        target_wall = start_wall + expected_wall_elapsed
        now = time.perf_counter()
        
        if target_wall > now:
            time.sleep(target_wall - now)
            
        yield incoming
