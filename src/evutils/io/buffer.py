import numpy as np

from ..types import EventArray, TriggerArray
from ._native_evt import EventSoABuffers, TriggerSoABuffers


class EventAccumulator():
    """Reused struct-of-arrays staging buffer that native decoders fill in place.

    The whole point is to avoid a copy on the read path: instead of a decoder
    parsing into its own buffer and the reader copying that into a ring, the
    decoder's parser writes *directly* into this accumulator's storage (via
    :meth:`prepare` + ``decoder.parse_step``). Consumed events at the front are
    reclaimed by :meth:`_rotate` (moving only the small unconsumed remainder), so
    the backing arrays are allocated once and never re-faulted. Only the final
    window handed to the caller is copied (:meth:`slice_copy`).

    Timestamps are stored as ``uint64`` (matching the native ``timestamp64_t``)
    and exposed as ``int64`` via a zero-copy ``.view`` (values are positive and
    in range).
    """

    def __init__(self, capacity: int):
        self._capacity = int(capacity)
        self._buf = EventSoABuffers(self._capacity)
        self._tr_capacity = max(self._capacity // 16, 1)
        self._tr = TriggerSoABuffers(self._tr_capacity)
        self._start = 0  # events before this index are consumed
        self._tr_start = 0

    def __len__(self) -> int:
        return self._buf.size - self._start

    def t_window(self) -> np.ndarray:
        """int64 view of the currently unconsumed timestamps (zero-copy)."""
        s, e = self._start, self._buf.size
        return self._buf.t[s:e].view(np.int64)

    def t_window_tr(self) -> np.ndarray:
        """int64 view of the currently unconsumed trigger timestamps (zero-copy)."""
        s, e = self._tr_start, self._tr.size
        return self._tr.t[s:e].view(np.int64)

    def _rotate(self) -> None:
        """Move the unconsumed remainder to the front, freeing consumed space."""
        s = self._start
        if s > 0:
            b = self._buf
            n = b.size - s
            if n > 0:
                b.t[:n] = b.t[s:b.size]
                b.x[:n] = b.x[s:b.size]
                b.y[:n] = b.y[s:b.size]
                b.p[:n] = b.p[s:b.size]
            b.c.size = n
            self._start = 0

        s_tr = self._tr_start
        if s_tr > 0:
            t = self._tr
            n_tr = t.size - s_tr
            if n_tr > 0:
                t.t[:n_tr] = t.t[s_tr:t.size]
                t.id[:n_tr] = t.id[s_tr:t.size]
                t.p[:n_tr] = t.p[s_tr:t.size]
            t.c.size = n_tr
            self._tr_start = 0

    def prepare(self, step: int):
        """Ready the buffer for the decoder to append up to ``step`` events, and
        return ``(events_soa, triggers_soa)`` for ``decoder.parse_step``.

        Rotates out consumed events if there is not enough tail room, then caps
        the SoA capacity the parser sees to ``size + step`` so a single step does
        not overshoot the requested window by more than one step's worth."""
        b = self._buf
        t = self._tr
        if self._capacity - b.size < step or t.capacity - t.size < step // 16:
            self._rotate()
        b.c.capacity = min(self._capacity, b.size + step)
        t.c.capacity = t.capacity
        return b, t

    def append(self, data: EventArray, triggers=None) -> None:
        """Copy an EventArray in (fallback path for non-native decoders)."""
        n = len(data)
        if n > 0:
            b = self._buf
            if self._capacity - b.size < n:
                self._rotate()
            if self._capacity - b.size < n:
                raise ValueError(
                    f"EventAccumulator full: can't append {n} ({self._capacity - b.size} free)"
                )
            e = b.size
            b.t[e:e + n] = data.t
            b.x[e:e + n] = data.x
            b.y[e:e + n] = data.y
            b.p[e:e + n] = data.p
            b.c.size = e + n

        if triggers is not None and len(triggers) > 0:
            n_tr = len(triggers)
            t = self._tr
            if t.capacity - t.size < n_tr:
                self._rotate()
            if t.capacity - t.size < n_tr:
                t.grow(t.size + n_tr)
            e_tr = t.size
            t.t[e_tr:e_tr + n_tr] = triggers.t
            t.id[e_tr:e_tr + n_tr] = triggers.id
            t.p[e_tr:e_tr + n_tr] = triggers.p
            t.c.size = e_tr + n_tr

    def slice_copy(self, k: int, tr_k: int = 0):
        """Return an independent copy of the first ``k`` unconsumed events and
        advance past them."""
        s = self._buf
        i = self._start
        out = EventArray(
            s.t[i:i + k].view(np.int64), s.x[i:i + k], s.y[i:i + k], s.p[i:i + k]
        ).copy()
        self._start += k

        t = self._tr
        j = self._tr_start
        out_tr = TriggerArray(
            t.t[j:j + tr_k].view(np.int64), t.p[j:j + tr_k], t.id[j:j + tr_k]
        ).copy()
        self._tr_start += tr_k

        return out, out_tr

    def reset(self) -> None:
        self._buf.c.size = 0
        self._tr.c.size = 0
        self._start = 0
        self._tr_start = 0

