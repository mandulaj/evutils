"""Core data types for event streams.

Defines the structured NumPy dtypes used throughout evutils — ``Events``
(timestamp, x, y, polarity) and ``Triggers`` (timestamp, polarity, id) —
together with small helpers for checking event arrays.
"""


import ctypes
from typing import Any

import numpy as np

__all__ = ['Event_dtype', 'Trigger_dtype', 'Event', 'Events', 'IndexedEvents', 'EventArray', 'TriggerArray', 'is_monotonically_increasing']


#: A structured numpy dtype for event data.
#:
#: Fields:
#:
#: - `t` (np.int64): Timestamp of the event (us).
#: - `x` (np.uint16): X-coordinate.
#: - `y` (np.uint16): Y-coordinate.
#: - `p` (np.uint8): Polarity (0: off, 1: on).
Event_dtype = np.dtype([('t', np.int64), ('x', np.uint16), ('y', np.uint16), ('p', np.uint8)])


#: A structured numpy dtype for trigger data.
#:
#: Fields:
#:
#: - `t` (np.int64): Timestamp of the event (us).
#: - `p` (np.uint8):  Polarity (0: off, 1: on).
#: - `id` (np.uint8): Identifier.
Trigger_dtype = np.dtype([('t', np.int64), ('p', np.uint8), ('id', np.uint8)])


class Event(ctypes.Structure):
    """Ctypes structure representing an event.

    Fields
    ------
    t : ctypes.c_int64
        Timestamp of the event (us).
    x : ctypes.c_uint16
        X-coordinate.
    y : ctypes.c_uint16
        Y-coordinate.
    p : ctypes.c_uint8
        Polarity (0: off, 1: on).
    """

    _fields_ = [("t", ctypes.c_int64),
                ("x", ctypes.c_uint16),
                ("y", ctypes.c_uint16),
                ("p", ctypes.c_uint8)]



def is_monotonically_increasing(events: np.ndarray) -> bool:
    """Checks if the event ts is monotonically increasing.

    Parameters
    ----------
    events : np.ndarray
        Array of events with a 't' field for timestamps.

    Returns
    -------
    bool
        True if timestamps are monotonically increasing, False otherwise.

    """
    return bool(np.all(np.diff(events['t']) >= 0))



class EventArray:
    """A container for storing event data in a struct-of-arrays (SoA) layout.

    The four fields ``t``, ``x``, ``y`` and ``p`` are kept as separate
    contiguous numpy arrays. This is the native layout of the C parser and
    avoids the padding of the packed :data:`Event_dtype` struct.

    Both attribute access (``events.t``) and key access (``events['t']``) return
    the underlying column, so most code written for structured arrays keeps
    working. ``np.asarray(events)`` yields the array-of-structures form (see
    :meth:`__array__`), which lets EventArray flow into code that still expects
    :data:`Event_dtype`.
    """

    __slots__ = ['t', 'x', 'y', 'p']
    _aos_dtype = Event_dtype

    def __init__(self, t: Any, x: Any, y: Any, p: Any) -> None:
        """Initializes the EventArray with columns.

        Parameters
        ----------
        t : array_like
            Timestamps of the events.
        x : array_like
            X-coordinates of the events.
        y : array_like
            Y-coordinates of the events.
        p : array_like
            Polarities of the events.

        """
        self.t = np.asarray(t, dtype=np.int64)
        self.x = np.asarray(x, dtype=np.uint16)
        self.y = np.asarray(y, dtype=np.uint16)
        self.p = np.asarray(p, dtype=np.uint8)

    def __getitem__(self, key: Any) -> Any:
        if isinstance(key, str):
            return getattr(self, key)

        return EventArray(self.t[key], self.x[key], self.y[key], self.p[key])

    def __len__(self) -> int:
        return len(self.t)

    def __repr__(self) -> str:
        return f"EventArray(n={len(self)})"

    def copy(self) -> "EventArray":
        """Return a deep copy with independent column arrays."""
        return EventArray(self.t.copy(), self.x.copy(), self.y.copy(), self.p.copy())

    @classmethod
    def empty(cls) -> "EventArray":
        """Return an empty EventArray with correctly-typed (zero-length) columns."""
        return cls(
            np.empty(0, dtype=np.int64),
            np.empty(0, dtype=np.uint16),
            np.empty(0, dtype=np.uint16),
            np.empty(0, dtype=np.uint8),
        )

    @classmethod
    def from_aos(cls, aos_array: Any) -> "EventArray":
        """Constructs an EventArray from an array of structures (AoS) numpy array.

        Parameters
        ----------
        aos_array : np.ndarray
            Structured numpy array of events.

        Returns
        -------
        EventArray
            The constructed EventArray in struct-of-arrays format.

        """
        return cls(
            np.ascontiguousarray(aos_array['t']),
            np.ascontiguousarray(aos_array['x']),
            np.ascontiguousarray(aos_array['y']),
            np.ascontiguousarray(aos_array['p'])
        )

    def to_aos(self) -> np.ndarray:
        """Converts the EventArray to an array of structures (AoS) numpy array.

        Returns
        -------
        np.ndarray
            Structured numpy array of events.

        """
        aos_array = np.empty(len(self), dtype=self._aos_dtype)
        aos_array['t'] = self.t
        aos_array['x'] = self.x
        aos_array['y'] = self.y
        aos_array['p'] = self.p
        return aos_array

    def __array__(self, dtype: Any = None, copy: Any = None) -> np.ndarray:
        """Numpy interop: ``np.asarray(events)`` returns the AoS structured array.

        This is the automatic SoA->AoS bridge for code (and tests) that still
        operate on :data:`Event_dtype` arrays.

        Parameters
        ----------
        dtype : data-type, optional
            Desired data type for the array.
        copy : bool, optional
            Whether to copy the data (ignored).

        Returns
        -------
        np.ndarray
            The structured array of events.

        """
        aos = self.to_aos()
        if dtype is not None:
            return aos.astype(dtype)
        return aos


class TriggerArray:
    """A container for storing trigger data in a struct-of-arrays (SoA) layout."""

    __slots__ = ['t', 'p', 'id']
    _aos_dtype = Trigger_dtype

    def __init__(self, t: Any, p: Any, id: Any) -> None:
        """Initializes the TriggerArray with columns.

        Parameters
        ----------
        t : array_like
            Timestamps of the triggers.
        p : array_like
            Polarities of the triggers.
        id : array_like
            Identifiers of the triggers.

        """
        self.t = np.asarray(t, dtype=np.int64)
        self.p = np.asarray(p, dtype=np.uint8)
        self.id = np.asarray(id, dtype=np.uint8)

    def __getitem__(self, key: Any) -> Any:
        if isinstance(key, str):
            return getattr(self, key)
        return TriggerArray(self.t[key], self.p[key], self.id[key])

    def __len__(self) -> int:
        return len(self.t)

    def __repr__(self) -> str:
        return f"TriggerArray(n={len(self)})"

    def copy(self) -> "TriggerArray":
        """Return a deep copy with independent column arrays.

        Returns
        -------
        TriggerArray
            A deep copy of the trigger array.

        """
        return TriggerArray(self.t.copy(), self.p.copy(), self.id.copy())

    @classmethod
    def empty(cls) -> "TriggerArray":
        """Return an empty TriggerArray with correctly-typed (zero-length) columns.

        Returns
        -------
        TriggerArray
            An empty trigger array.

        """
        return cls(
            np.empty(0, dtype=np.int64),
            np.empty(0, dtype=np.uint8),
            np.empty(0, dtype=np.uint8),
        )

    def to_aos(self) -> np.ndarray:
        """Converts the TriggerArray to an array of structures (AoS) numpy array.

        Returns
        -------
        np.ndarray
            Structured numpy array of triggers.

        """
        aos_array = np.empty(len(self), dtype=self._aos_dtype)
        aos_array['t'] = self.t
        aos_array['p'] = self.p
        aos_array['id'] = self.id
        return aos_array

    def __array__(self, dtype: Any = None, copy: Any = None) -> np.ndarray:
        """Numpy interop: ``np.asarray(triggers)`` returns the AoS structured array.

        Parameters
        ----------
        dtype : data-type, optional
            Desired data type for the array.
        copy : bool, optional
            Whether to copy the data (ignored).

        Returns
        -------
        np.ndarray
            The structured array of triggers.

        """
        aos = self.to_aos()
        if dtype is not None:
            return aos.astype(dtype)
        return aos