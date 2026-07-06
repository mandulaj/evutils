"""Core data types for event streams.

Defines the structured NumPy dtypes used throughout evutils — ``Events``
(timestamp, x, y, polarity) and ``Triggers`` (timestamp, polarity, id) —
together with small helpers for checking event arrays.
"""


import ctypes
from typing import Any, TypeVar

import numpy as np

__all__ = ['Event_dtype', 'Trigger_dtype', 'Event', 'EventArray', 'TriggerArray', 'is_monotonically_increasing']


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



_S = TypeVar("_S", bound="SoaArray")


class SoaArray:
    """Abstract base class for struct-of-arrays (SoA) layout.

    Examples
    --------
    >>> import numpy as np
    >>> from evutils.types import EventArray
    >>> events = EventArray(t=[1, 2], x=[10, 20], y=[30, 40], p=[1, 0])
    >>> float(np.mean(events.x))  # SoA layout allows fast operations on single columns
    15.0
    >>> events.to_numpy()  # doctest: +SKIP
    array([(1, 10, 30, 1), (2, 20, 40, 0)],
          dtype=[('t', '<i8'), ('x', '<u2'), ('y', '<u2'), ('p', 'u1')])
    """

    __slots__ = ()
    _aos_dtype: np.dtype
    _fields: tuple[str, ...]

    def __getitem__(self, key: Any) -> Any:
        if isinstance(key, str):
            return getattr(self, key)
        
        # When indexing a single element, return a NumPy void record to match AoS behaviour exactly.
        if isinstance(key, (int, np.integer)):
            record = np.empty((), dtype=self._aos_dtype)
            for f in self._fields:
                record[f] = getattr(self, f)[key]
            return record[()]  # Returns a scalar np.void

        # Otherwise, slice all columns and return a new SoA array
        sliced_args = {f: getattr(self, f)[key] for f in self._fields}
        return self.__class__(**sliced_args)

    def __len__(self) -> int:
        return len(getattr(self, self._fields[0]))

    def __repr__(self) -> str:
        n = len(self)
        name = self.__class__.__name__
        if n == 0:
            return f"{name}(empty)"
            
        if n <= 10:
            return f"{name}(n={n}):\n{self.to_aos()}"
            
        # Slice before AoS conversion for speed
        head_str = str(self[:3].to_aos()).rstrip(']')
        tail_str = str(self[-3:].to_aos()).lstrip('[')
        
        return f"{name}(n={n}):\n{head_str}\n ...\n  {tail_str}]"

    def copy(self: _S) -> _S:
        """Return a deep copy with independent column arrays."""
        copied_args = {f: getattr(self, f).copy() for f in self._fields}
        return self.__class__(**copied_args)

    @classmethod
    def empty(cls: 'type[_S]') -> _S:
        """Return an empty SoA array with correctly-typed (zero-length) columns."""
        args = {f: np.empty(0, dtype=cls._aos_dtype[f]) for f in cls._fields}
        return cls(**args)

    @classmethod
    def from_aos(cls: 'type[_S]', aos_array: np.ndarray) -> _S:
        """Constructs a SoA array from an array of structures (AoS) numpy array."""
        args = {f: np.ascontiguousarray(aos_array[f]) for f in cls._fields}
        return cls(**args)

    def to_aos(self) -> np.ndarray:
        """Converts the SoA array to an array of structures (AoS) numpy array."""
        aos_array = np.empty(len(self), dtype=self._aos_dtype)
        for f in self._fields:
            aos_array[f] = getattr(self, f)
        return aos_array

    def to_numpy(self) -> np.ndarray:
        """Converts the SoA array to a structured numpy array. Alias for to_aos."""
        return self.to_aos()

    def __array__(self, dtype: Any = None, copy: Any = None) -> np.ndarray:
        """Numpy interop: ``np.asarray(arr)`` returns the AoS structured array."""
        aos = self.to_aos()
        if dtype is not None:
            return aos.astype(dtype)
        return aos


class EventArray(SoaArray):
    """A container for storing event data in a struct-of-arrays (SoA) layout.

    The four fields ``t``, ``x``, ``y`` and ``p`` are kept as separate
    contiguous numpy arrays. This is the native layout of the C parser and
    avoids the padding of the packed :data:`Event_dtype` struct.

    Both attribute access (``events.t``) and key access (``events['t']``) return
    the underlying column, so most code written for structured arrays keeps
    working. ``np.asarray(events)`` yields the array-of-structures form (see
    :meth:`__array__`), which lets EventArray flow into code that still expects
    :data:`Event_dtype`.

    Examples
    --------
    >>> from evutils.types import EventArray
    >>> events = EventArray(t=[100, 150], x=[10, 20], y=[30, 40], p=[1, 0])
    >>> events.t
    array([100, 150])
    >>> events.x
    array([10, 20], dtype=uint16)
    >>> events.y
    array([30, 40], dtype=uint16)
    >>> events.p
    array([1, 0], dtype=uint8)
    >>> events[:10]
    EventArray(n=2):
    [(100, 10, 30, 1) (150, 20, 40, 0)]
    >>> events[0]  # doctest: +SKIP
    np.void((100, 10, 30, 1), dtype=[('t', '<i8'), ('x', '<u2'), ('y', '<u2'), ('p', 'u1')])
    """

    __slots__ = ['t', 'x', 'y', 'p']
    _aos_dtype = Event_dtype
    _fields = ('t', 'x', 'y', 'p')

    def __init__(self, t: Any, x: Any, y: Any, p: Any) -> None:
        self.t = np.asarray(t, dtype=np.int64)
        self.x = np.asarray(x, dtype=np.uint16)
        self.y = np.asarray(y, dtype=np.uint16)
        self.p = np.asarray(p, dtype=np.uint8)


class TriggerArray(SoaArray):
    """A container for storing trigger data in a struct-of-arrays (SoA) layout.

    Examples
    --------
    >>> from evutils.types import TriggerArray
    >>> triggers = TriggerArray(t=[1000, 2000], p=[1, 0], id=[0, 1])
    >>> triggers.t
    array([1000, 2000])
    >>> triggers.p
    array([1, 0], dtype=uint8)
    >>> triggers.id
    array([0, 1], dtype=uint8)
    """

    __slots__ = ['t', 'p', 'id']
    _aos_dtype = Trigger_dtype
    _fields = ('t', 'p', 'id')

    def __init__(self, t: Any, p: Any, id: Any) -> None:
        self.t = np.asarray(t, dtype=np.int64)
        self.p = np.asarray(p, dtype=np.uint8)
        self.id = np.asarray(id, dtype=np.uint8)