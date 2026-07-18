"""Per-mode window strategies for the decomposed :class:`EventReader` (V2).

Each :class:`WindowStrategy` produces the next window from a shared
:class:`~evutils.io.v2.context.ReadContext`. The concrete strategies absorb the
V1 monolith's read methods behaviour-for-behaviour (the numeric logic is pinned
by tests and ported verbatim):

* :class:`DeltaTStrategy` -- the two delta_t fast paths (dedicated C parser or
  ``searchsorted`` + overshoot carry), sharing the ``_finalize`` tail.
* :class:`NEventsStrategy` -- zero-copy native fill, ``read_chunk`` passthrough.
* :class:`MixedStrategy` -- the general accumulator while-loop (both cutoffs).
* :class:`AllStrategy` -- drain-to-EOF, plus the full-payload ``read_all``.

``select_strategy`` reproduces the dispatch guards from V1's ``_read_impl``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import numpy as np

from ...types import EventArray, TriggerArray
from .._native_core import (EVUTILS_PARSE_OUTPUT_FULL, EVUTILS_PARSE_WINDOW_DONE,
                            EventSoABuffers, TriggerSoABuffers, events_view)

if TYPE_CHECKING:
    from .context import ReadContext


class WindowStrategy(ABC):
    """Produces successive windows over a :class:`ReadContext`."""

    def __init__(self, ctx: "ReadContext"):
        self.ctx = ctx

    @abstractmethod
    def next_window(self, delta_t: "int | None" = None, n_events: "int | None" = None
                    ) -> "EventArray | tuple[EventArray, TriggerArray]":
        """Decode and return the next window (empty at EOF)."""
        raise NotImplementedError

    def reset(self) -> None:
        """Hook for strategy-local state reset (context state is reset by the
        reader). No strategy keeps state outside the context, so this is a
        no-op by default."""


# --------------------------------------------------------------------------- #
# delta_t
# --------------------------------------------------------------------------- #
class DeltaTStrategy(WindowStrategy):
    """One ``delta_t`` window per call, zero-copy.

    Picks its sub-path on ``decoder._has_delta_t_parser``: the dedicated C
    parser (one GIL-free call per window, no boundary search) when present, else
    the generic ``searchsorted`` + overshoot-carry path. Both share
    :meth:`_finalize`.
    """

    def next_window(self, delta_t: "int | None" = None, n_events: "int | None" = None
                    ) -> "EventArray":
        dt = delta_t if delta_t is not None else self.ctx.delta_t
        if self.ctx.decoder._has_delta_t_parser:
            return self._read_c(dt)
        return self._read_fast(dt)

    # ---- shared tail (ported from _finalize_delta_t_window) ---------- #
    def _finalize(self, out: "EventSoABuffers", delta_t: int, count_cut: bool,
                  resume_ts: "int | None" = None) -> "EventArray":
        ctx = self.ctx
        size = out.size
        # Track the largest window seen so the next buffer is pre-sized for it.
        ctx.dt_est = max(ctx.dt_est, size)
        if count_cut:
            # Count cutoff: continue the same time window on the next call.
            if resume_ts is not None:
                ctx.current_ts = resume_ts
        else:
            ctx.current_ts += delta_t
        ctx.n_read_events += size

        output = events_view(out)
        if ctx.normalize_ts:
            output.t -= ctx.first_ts - ctx.start_ts
        return output

    # ---- searchsorted + carry path (ported from _read_delta_t_fast) -- #
    def _read_fast(self, delta_t: int) -> "EventArray":
        ctx = self.ctx
        dec = ctx.decoder
        step = ctx.dt_step
        n_cap = ctx.n_events
        tr = ctx.dt_trigger_sink()

        carry = ctx.dt_carry
        k = carry.size if carry is not None else 0
        need = max(ctx.dt_est * 5 // 4 + step, k + step)
        out = ctx.acquire_window_buffer(need)
        if k:
            out.t[:k] = carry.t[:k]
            out.x[:k] = carry.x[:k]
            out.y[:k] = carry.y[:k]
            out.p[:k] = carry.p[:k]
            out.c.size = k
        ctx.dt_carry = None

        # Anchor the stream's first timestamp / window origin on the very first
        # decoded event (only reachable on the first call, when there is no carry).
        if not ctx.anchored and out.size == 0:
            while out.size == 0 and not dec.is_eof():
                ctx.grow_out(out, step)
                tr.reset()
                dec.parse_step(out, tr)  # type: ignore[attr-defined]
            if out.size == 0:
                ctx.eof = True
                return EventArray.empty()
            ctx.anchor_first_ts(int(out.t[0]))

        end_ts = ctx.current_ts + delta_t

        # Decode until the buffer holds an event at/after the time boundary, the
        # count cap is exceeded, or the stream ends.
        while not dec.is_eof():
            n = out.size
            if n and (int(out.t[n - 1]) >= end_ts or n > n_cap):
                break
            ctx.grow_out(out, step)
            tr.reset()
            added = dec.parse_step(out, tr)  # type: ignore[attr-defined]
            if added == 0 and dec.is_eof():
                break

        size = out.size
        t = out.t[:size].view(np.int64)
        idx = int(np.searchsorted(t, end_ts)) if size else 0
        count_cut = idx > n_cap
        if count_cut:
            idx = n_cap

        # Everything from idx on is overshoot: copy it (small) for the next call.
        if idx < size:
            rem = size - idx
            c = EventSoABuffers(rem)
            c.t[:rem] = out.t[idx:size]
            c.x[:rem] = out.x[idx:size]
            c.y[:rem] = out.y[idx:size]
            c.p[:rem] = out.p[idx:size]
            c.c.size = rem
            ctx.dt_carry = c

        out.c.size = idx
        # EOF only once the stream is drained *and* no overshoot remains, else the
        # final carried events would never be emitted (is_eof() stops iteration).
        ctx.eof = dec.is_eof() and ctx.dt_carry is None
        return self._finalize(
            out, delta_t, count_cut,
            resume_ts=int(t[idx]) if count_cut else None)

    # ---- dedicated C parser path (ported from _read_delta_t_c) ------- #
    def _read_c(self, delta_t: int) -> "EventArray":
        ctx = self.ctx
        dec = ctx.decoder
        n_cap = ctx.n_events
        tr = ctx.dt_trigger_sink()

        # Anchor the window origin on the first event once, then rewind so the C
        # parser redoes window 0 from the start with the right end_ts.
        if not ctx.anchored:
            first = ctx.peek_first_ts()
            if first is None:
                ctx.eof = True
                return EventArray.empty()
            ctx.anchor_first_ts(first)
            dec.reset()
            ctx.eof = False

        end_ts = ctx.current_ts + delta_t

        out = ctx.acquire_window_buffer(
            max(ctx.dt_est * 5 // 4 + ctx.dt_step, ctx.dt_step))
        # The n_events safety cap (max_events in delta_t mode) is enforced by
        # clamping the capacity the C parser sees, so a single C call can never
        # decode past it.
        out.c.capacity = min(out.capacity, n_cap)
        count_cut = False
        while True:
            tr.reset()
            _, status = dec.parse_step_delta_t(out, tr, end_ts)  # type: ignore[attr-defined]
            if status == EVUTILS_PARSE_WINDOW_DONE:
                break
            if dec.is_eof():
                break
            if status == EVUTILS_PARSE_OUTPUT_FULL:
                if out.c.capacity >= n_cap:
                    # Hit the max_events cap: emit what we have and continue
                    # this same time window on the next call.
                    count_cut = True
                    break
                # Window not finished but the buffer filled: grow and continue.
                new_cap = min(max(out.size + ctx.dt_step,
                                  int(out.c.capacity * 1.5) + 1), n_cap)
                out.grow(new_cap)          # realloc only if backing too small
                out.c.capacity = new_cap   # parser-visible cap (recycled slots)
            # else EVUTILS_PARSE_OK: input drained this call; loop (tail -> EOF).

        # On a normal window advance the clock; on a count-cut stay in the same
        # time window so the next call continues it (matches the accumulator path).
        ctx.eof = dec.is_eof()
        return self._finalize(out, delta_t, count_cut)


# --------------------------------------------------------------------------- #
# n_events
# --------------------------------------------------------------------------- #
class NEventsStrategy(WindowStrategy):
    """Pure ``n_events`` window, zero-copy.

    Native zero-copy fill for exact-capacity decoders (EVT2/EVT4/DAT), or a
    ``read_chunk`` passthrough for decoders whose chunks are already independent
    and bounded (NPZ/CSV).
    """

    def next_window(self, delta_t: "int | None" = None, n_events: "int | None" = None
                    ) -> "EventArray":
        ctx = self.ctx
        dec = ctx.decoder
        n = n_events if n_events is not None else ctx.n_events
        if ctx.native_fill and dec._exact_window:
            return self._read_fast(n)
        return self._read_readchunk(n)

    # ---- zero-copy native fill (ported from _read_n_events_fast) ----- #
    def _read_fast(self, n_events: int) -> "EventArray":
        ctx = self.ctx
        dec = ctx.decoder
        out = EventSoABuffers(n_events)
        out.c.capacity = n_events
        tr = TriggerSoABuffers(max(n_events // 16, 1024))
        while out.size < n_events and not dec.is_eof():
            tr.reset()
            before = out.size
            dec.parse_step(out, tr)  # type: ignore[attr-defined]
            if out.size == before and dec.is_eof():
                break
        ctx.eof = dec.is_eof()

        if out.size == 0:
            return EventArray.empty()
        if not ctx.anchored:
            ctx.anchor_first_ts(int(out.t[0]))
        ctx.n_read_events += out.size

        output = events_view(out)
        if ctx.normalize_ts:
            output.t -= ctx.first_ts - ctx.start_ts
        return output

    # ---- read_chunk passthrough (ported from _read_n_events_readchunk) #
    def _read_readchunk(self, n_events: int) -> "EventArray":
        ctx = self.ctx
        dec = ctx.decoder
        chunk = dec.read_chunk(n_events_hint=n_events)
        if isinstance(chunk, tuple):
            chunk = chunk[0]
        # A full window is exactly n_events; anything short means the stream is
        # drained. (Do not use dec.is_eof(): for buffered decoders like CSV it
        # signals "input exhausted" while read_chunk still has buffered events.)
        ctx.eof = len(chunk) < n_events
        if len(chunk) == 0:
            return EventArray.empty()
        if ctx.n_read_events == 0:
            ctx.first_ts = int(chunk.t[0])
            ctx.current_ts = ctx.first_ts
        ctx.n_read_events += len(chunk)
        if ctx.normalize_ts:
            chunk.t -= ctx.first_ts - ctx.start_ts  # chunk is independent
        return chunk


# --------------------------------------------------------------------------- #
# accumulator loop (mixed + all share this)
# --------------------------------------------------------------------------- #
def accumulator_window(ctx: "ReadContext", delta_t: "int | None",
                       n_events: "int | None", drain: bool
                       ) -> "EventArray | tuple[EventArray, TriggerArray]":
    """The general staging-accumulator window (ported from ``_read_impl`` tail).

    Both cutoffs (time + count) are evaluated together; whichever falls earlier
    wins. ``drain=True`` disables the cutoffs and reads to EOF ("all" mode). Any
    pending delta_t fast-path overshoot is folded in first.
    """
    acc = ctx.ensure_accumulator()

    # If a delta_t fast path ran earlier and left an overshoot carry, fold it in
    # first: those events precede anything the decoder emits next.
    ctx.flush_dt_carry()

    # Override the parameters if they are specified.
    if delta_t is None:
        delta_t = ctx.delta_t
    if n_events is None:
        n_events = ctx.n_events

    # Establish the first timestamp once, at the very start of the stream.
    if not ctx.anchored and len(acc) == 0:
        if _pull(ctx, acc, delta_t, n_events) == 0:
            ctx.eof = True
            if ctx.read_external_triggers:
                return EventArray.empty(), TriggerArray.empty()
            return EventArray.empty()
        ctx.anchor_first_ts(int(acc.t_window()[0]))

    start_ts: int = ctx.current_ts
    end_ts: int = start_ts + delta_t  # Final end_ts if we reach delta_t
    end_idx: int = len(acc)

    tr_end_idx: int = acc._tr.size - acc._tr_start

    # Gather events until we hit the n_events count, the delta_t time window, or
    # the end of the stream. Both cutoffs are evaluated together.
    while True:
        t = acc.t_window()
        time_ready = not drain and len(t) > 0 and t[-1] >= end_ts
        count_ready = not drain and len(acc) > n_events

        if time_ready or count_ready:
            time_idx = int(np.searchsorted(t, end_ts)) if time_ready else len(acc) + 1
            tr_t = acc.t_window_tr()
            if count_ready and time_idx > n_events:
                # n_events cutoff comes first
                end_idx = n_events
                ctx.current_ts = int(t[n_events])
                tr_end_idx = int(np.searchsorted(tr_t, ctx.current_ts, side='left'))
            else:
                # delta_t cutoff comes first (ties go to the time window)
                end_idx = time_idx
                ctx.current_ts += delta_t
                tr_end_idx = int(np.searchsorted(tr_t, end_ts, side='left'))
            break

        # Not enough buffered yet: pull more from the decoder.
        if _pull(ctx, acc, delta_t, n_events) == 0:
            ctx.eof = True
            # Neither cutoff met, so the whole remaining buffer is the slice.
            end_idx = len(acc)
            tr_end_idx = acc._tr.size - acc._tr_start
            break

    # Copy out the window (independent) and advance past it.
    output, output_tr = acc.slice_copy(end_idx, tr_end_idx)
    ctx.n_read_events += end_idx

    if ctx.normalize_ts:
        output.t -= ctx.first_ts - ctx.start_ts
        if len(output_tr) > 0:
            output_tr.t -= ctx.first_ts - ctx.start_ts

    if ctx.read_external_triggers:
        return output, output_tr
    return output


def _pull(ctx: "ReadContext", acc, delta_t: int, n_events: int) -> int:
    """Pull more events into the accumulator (ported from ``_pull``).

    Returns the number added (0 => end of stream). Native decoders decode
    straight into the accumulator's storage (no copy); others have their
    ``read_chunk`` output appended.
    """
    dec = ctx.decoder
    if ctx.native_fill:
        while True:
            if dec.is_eof():
                return 0
            ev, tr = acc.prepare(ctx.step)
            added = dec.parse_step(ev, tr)  # type: ignore[attr-defined]
            if added > 0:
                return int(added)
            if dec.is_eof():
                return 0
            # else: consumed only state/timing words; step again.
    chunk = dec.read_chunk(delta_t, n_events)
    if isinstance(chunk, tuple):
        chunk, triggers = chunk
    else:
        triggers = None
    if len(chunk) == 0 and (triggers is None or len(triggers) == 0):
        return 0
    acc.append(chunk, triggers)
    return int(len(chunk)) if len(chunk) > 0 else 1


class MixedStrategy(WindowStrategy):
    """The general accumulator window (both cutoffs). Handles ``mode='mixed'``
    and any per-call override that falls off a fast path."""

    def next_window(self, delta_t: "int | None" = None, n_events: "int | None" = None
                    ) -> "EventArray | tuple[EventArray, TriggerArray]":
        return accumulator_window(self.ctx, delta_t, n_events, drain=False)


class AllStrategy(WindowStrategy):
    """Drain-to-EOF ("all" mode). Also owns the full-payload ``read_all``."""

    def next_window(self, delta_t: "int | None" = None, n_events: "int | None" = None
                    ) -> "EventArray | tuple[EventArray, TriggerArray]":
        # "all" mode with per-call overrides degrades to a windowed read (drain
        # is only taken when both overrides are None -- select_strategy enforces
        # that, so here drain is always True).
        return accumulator_window(self.ctx, delta_t, n_events, drain=True)

    def read_all(self) -> "EventArray | tuple[EventArray, TriggerArray]":
        """Decode and return every remaining event at once (ported from
        ``EventReader.read_all``, minus the batch/sensor wrapping the facade
        applies)."""
        ctx = self.ctx
        # Fold in any delta_t fast-path overshoot so it is prepended below.
        ctx.flush_dt_carry()

        _out = ctx.decoder.read_all()
        if ctx.read_external_triggers:
            if isinstance(_out, tuple):
                out, out_tr = _out
            else:
                out = _out
                out_tr = TriggerArray.empty()
        else:
            out = _out  # type: ignore

        # Prepend anything already buffered by prior read() calls (rare).
        acc = ctx.accumulator
        if acc is not None and (len(acc) > 0 or acc._tr.size - acc._tr_start > 0):
            buffered, buffered_tr = acc.slice_copy(len(acc), acc._tr.size - acc._tr_start)
            if len(out) == 0:
                out = buffered
            elif len(buffered) > 0:
                out = EventArray(
                    np.concatenate([buffered.t, out.t]),
                    np.concatenate([buffered.x, out.x]),
                    np.concatenate([buffered.y, out.y]),
                    np.concatenate([buffered.p, out.p]),
                )

            if ctx.read_external_triggers:
                if len(out_tr) == 0:
                    out_tr = buffered_tr
                elif len(buffered_tr) > 0:
                    out_tr = TriggerArray(
                        np.concatenate([buffered_tr.t, out_tr.t]),
                        np.concatenate([buffered_tr.p, out_tr.p]),
                        np.concatenate([buffered_tr.id, out_tr.id]),
                    )

        ctx.eof = True
        ctx.n_read_events += len(out)

        if ctx.normalize_ts and len(out) > 0:
            shift = int(out.t[0]) - ctx.start_ts
            out.t -= shift
            if ctx.read_external_triggers and len(out_tr) > 0:
                out_tr.t -= shift

        if ctx.read_external_triggers:
            return out, out_tr
        return out


# --------------------------------------------------------------------------- #
# strategy selection (ported from _read_impl's dispatch guards)
# --------------------------------------------------------------------------- #
def select_strategy(ctx: "ReadContext", mode: str, delta_t: "int | None",
                    n_events: "int | None") -> WindowStrategy:
    """Choose the strategy for a single read, reproducing V1's ``_read_impl``
    dispatch (fast paths gated on mode, per-call overrides, triggers, buffer
    emptiness, and decoder capabilities)."""
    dec = ctx.decoder
    acc = ctx.accumulator
    buffer_empty = acc is None or len(acc) == 0

    # Fast path: pure n_events streaming, no time override, no triggers, nothing
    # buffered.
    if (mode == "n_events" and delta_t is None
            and not ctx.read_external_triggers and buffer_empty):
        if (ctx.native_fill and dec._exact_window) or dec._independent_windows:
            return NEventsStrategy(ctx)

    # Fast path: pure delta_t streaming with a native-fill decoder, no triggers,
    # nothing buffered.
    if (mode == "delta_t" and n_events is None
            and ctx.native_fill and not ctx.read_external_triggers
            and buffer_empty):
        return DeltaTStrategy(ctx)

    # "all" mode without per-call overrides drains to EOF.
    if mode == "all" and delta_t is None and n_events is None:
        return AllStrategy(ctx)

    # Everything else: the general accumulator loop.
    return MixedStrategy(ctx)
