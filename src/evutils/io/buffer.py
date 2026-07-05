import numpy as np

from ..types import EventArray


class EventRingBuffer():
    """A ring buffer for events stored in struct-of-arrays (SoA) layout.

    Each field (``t``, ``x``, ``y``, ``p``) is a separate contiguous array of
    ``size`` capacity. Appends and slices operate on :class:`EventArray`, so
    events flow decoder -> buffer -> reader output without ever being repacked
    into a padded structured array.
    """

    def __init__(self, size: int):
        self._size = size
        self._t = np.empty(size, dtype=np.int64)
        self._x = np.empty(size, dtype=np.uint16)
        self._y = np.empty(size, dtype=np.uint16)
        self._p = np.empty(size, dtype=np.uint8)
        self._start = 0
        self._end = 0

    def __len__(self) -> int:
        return self._end - self._start

    @property
    def capacity(self) -> int:
        return self._size - self._end

    def append(self, data: EventArray):
        n = len(data)
        if n > self.capacity:
            self.rotate()  # Try rotating to get more space
        if n > self.capacity:
            raise ValueError(
                f"Ring Buffer is full, can't append {n} elements when {self.capacity} space left."
            )

        e = self._end
        self._t[e:e + n] = data.t
        self._x[e:e + n] = data.x
        self._y[e:e + n] = data.y
        self._p[e:e + n] = data.p
        self._end += n

    def advance(self, items: int):
        if self._start + items > self._size:
            raise ValueError(f"Cant advance beyond the buffer size {self._end}+{items}<{self._size}")
        self._start += items

    def view(self) -> EventArray:
        """Zero-copy EventArray view of the currently-buffered events."""
        s, e = self._start, self._end
        return EventArray(self._t[s:e], self._x[s:e], self._y[s:e], self._p[s:e])

    def reset(self):
        self._start = 0
        self._end = 0

    def rotate(self):
        cur_len = len(self)
        s, e = self._start, self._end
        self._t[0:cur_len] = self._t[s:e]
        self._x[0:cur_len] = self._x[s:e]
        self._y[0:cur_len] = self._y[s:e]
        self._p[0:cur_len] = self._p[s:e]
        self._start = 0
        self._end = cur_len

    def _abs(self, idx: int) -> int:
        return self._start + idx if idx >= 0 else self._end + idx

    def __getitem__(self, key) -> EventArray:
        if isinstance(key, slice):
            # Translate slice bounds into absolute buffer indices.
            if key.start is None:
                start_idx = self._start
            else:
                start_idx = self._abs(key.start)

            if key.stop is None:
                end_idx = self._end
            else:
                end_idx = self._abs(key.stop)

            assert start_idx <= end_idx

            sl = slice(start_idx, end_idx, key.step)
            return EventArray(self._t[sl], self._x[sl], self._y[sl], self._p[sl])
        elif isinstance(key, int):
            idx = self._abs(key)
            return EventArray(self._t[idx], self._x[idx], self._y[idx], self._p[idx])
        else:
            raise ValueError(f"Unsupported key in Ringbuffer {type(key)}")

    def __repr__(self) -> str:
        return repr(self.view())
