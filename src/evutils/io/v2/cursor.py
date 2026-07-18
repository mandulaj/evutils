"""Random-access seeking for the decomposed :class:`EventReader` (V2).

``SeekCursor`` ports the monolith's ``seek`` / ``_seek_linear`` /
``_peek_stream_first_ts``: the decoder's fast-path ``seek()`` (index / binary
search) when the source is seekable or the decoder buffers the payload in
memory, else a linear iterate-and-drop fallback. On seek it re-anchors the
:class:`ReadContext` and resets the active windowing state; the public
:class:`~evutils.io.common.SeekResult` contract is unchanged.
"""

from __future__ import annotations

import io
from typing import TYPE_CHECKING

import numpy as np

from ...types import EventArray
from ..common import SeekResult

if TYPE_CHECKING:
    from ...types import TriggerArray
    from .context import ReadContext
    from .reader import EventReader

_EMPTY_EVENTS = EventArray.empty()


class SeekCursor:
    """Repositions the read cursor by timestamp or event index over a shared
    :class:`ReadContext`.

    Holds a back-reference to the owning reader for the byte source, the active
    prefetch guard, the pacer anchor, and the staging buffer -- all of which a
    seek must consult or reset.
    """

    def __init__(self, reader: "EventReader", ctx: "ReadContext"):
        self.reader = reader
        self.ctx = ctx
        self.last_seek: "SeekResult | None" = None

    def seek(self, t: "int | None" = None, n: "int | None" = None,
             relative: bool = False) -> SeekResult:
        """Reposition by timestamp (``t``) or event index (``n``); exactly one
        must be given. ``relative=True`` adds to the current cursor."""
        reader = self.reader
        ctx = self.ctx
        if not reader._is_initialized:
            reader.init()
        if (t is None) == (n is None):
            raise ValueError("seek() requires exactly one of t= or n=.")

        dec = ctx.decoder
        if not dec.SUPPORTS_SEEK:
            raise NotImplementedError(
                f"{dec.__class__.__name__} does not support seeking."
            )

        if reader._active_prefetch is not None:
            reader._active_prefetch.close()

        if relative:
            if t is not None:
                t = ctx.current_ts + t
            else:
                n = ctx.n_read_events + n

        # Anchor the normalization origin (first_ts) on the stream's *first*
        # event before the cursor moves, so normalize_ts is call-order
        # independent. Peeking rewinds the decoder, so it is skipped on
        # non-seekable sources (and when normalization is off).
        first_ts: "int | None" = None
        seekable = getattr(reader._source, "seekable", lambda: True)()
        can_fast_seek = seekable or dec._buffers_in_memory
        if not ctx.anchored and ctx.normalize_ts and can_fast_seek:
            first_ts = self._peek_stream_first_ts()

        ctx.eof = False
        reader._pacer.reset()
        ctx.dt_carry = None
        ctx.dt_est = ctx.step
        ctx.dt_slot_i = 0
        if ctx.accumulator is not None:
            ctx.accumulator.reset()

        if can_fast_seek:
            try:
                res, rem_ev, rem_tr = dec.seek(t=t, n=n)
            except (io.UnsupportedOperation, OSError):
                res, rem_ev, rem_tr = self._seek_linear(t, n)
        else:
            res, rem_ev, rem_tr = self._seek_linear(t, n)

        if len(rem_ev) > 0 or (rem_tr is not None and len(rem_tr) > 0):
            acc = ctx.ensure_accumulator()
            acc.append(rem_ev, rem_tr)

        if not ctx.anchored:
            ctx.first_ts = first_ts if first_ts is not None else res.ts
        ctx.n_read_events = res.index if res.index >= 0 else (int(n) if n is not None else 0)
        ctx.current_ts = res.ts
        ctx.anchored = True
        self.last_seek = SeekResult(ts=res.ts, index=ctx.n_read_events, eof=res.eof)
        return self.last_seek

    def _peek_stream_first_ts(self) -> "int | None":
        """Timestamp of the stream's very first event, leaving the decoder
        rewound to the start. Only called before any read on a fast-seekable
        source. Returns ``None`` for an empty stream."""
        ctx = self.ctx
        dec = ctx.decoder
        dec.reset()
        first: "int | None"
        if ctx.native_fill:
            first = ctx.peek_first_ts()
        else:
            chunk = dec.read_chunk()
            if isinstance(chunk, tuple):
                chunk = chunk[0]
            first = int(chunk.t[0]) if len(chunk) > 0 else None
        dec.reset()
        return first

    def _seek_linear(self, t: "int | None", n: "int | None"
                     ) -> "tuple[SeekResult, EventArray, TriggerArray | None]":
        """Fallback seek for non-seekable sources: iterate and drop to target."""
        ctx = self.ctx
        dec = ctx.decoder
        axis = "t" if t is not None else "n"
        target = t if t is not None else n

        behind = ((axis == "t" and target < ctx.current_ts)
                  or (axis == "n" and target < ctx.n_read_events))
        seen = 0
        if behind:
            dec.reset()
        else:
            seen = ctx.n_read_events

        landed_ts = target
        idx = target if axis == "n" else -1
        rem_ev = _EMPTY_EVENTS
        rem_tr = None

        while True:
            chunk = dec.read_chunk()
            if isinstance(chunk, tuple):
                chunk, tr_chunk = chunk
            else:
                tr_chunk = None

            if len(chunk) == 0:
                ctx.eof = True
                break
            if axis == "t":
                k = int(np.searchsorted(chunk.t, target, side="left"))
                hit = k < len(chunk)
            else:
                k = max(0, min(target - seen, len(chunk)))
                hit = target < seen + len(chunk)
            if hit:
                rem_ev = chunk[k:].copy()
                if tr_chunk is not None:
                    if axis == "t":
                        tr_k = int(np.searchsorted(tr_chunk.t, target, side="left"))
                    else:
                        tr_k = 0
                    rem_tr = tr_chunk[tr_k:].copy()

                landed_ts = int(rem_ev.t[0]) if len(rem_ev) > 0 else target
                idx = seen + k
                break
            seen += len(chunk)
        return SeekResult(ts=landed_ts, index=idx, eof=ctx.eof), rem_ev, rem_tr
