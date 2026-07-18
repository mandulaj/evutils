"""Shared read state for the decomposed :class:`EventReader` (V2).

``ReadContext`` gathers the cross-cutting state that the V1 monolith scattered
as ``self._*`` instance attributes onto ``EventReader``. Every strategy, the
seek cursor, and the pacer read and mutate this single object, so the coupling
that used to be implicit (the ``_dt_carry`` hand-off between the fast paths and
the accumulator loop; the ``_anchored`` / ``_first_ts`` sharing; the reuse ring)
is now explicit and owned in one place.

The numeric helpers here are ported verbatim (behaviour-for-behaviour) from the
monolith's private methods so the strategies keep bit-for-bit parity with V1.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from .._native_core import (EventSoABuffers, TriggerSoABuffers, events_view)
from ..buffer import EventAccumulator

if TYPE_CHECKING:
    from ..common import EventDecoder


@dataclass
class ReadContext:
    """Mutable cross-cutting state shared by every V2 read component.

    Holds the decoder, the normalization origin, the anchoring flags, the
    staging accumulator (lazy), the delta_t fast-path carry + reuse ring, and
    the running event counter -- i.e. exactly the ``self._*`` fields the V1
    monolith mutated across its read methods.
    """

    decoder: "EventDecoder"

    # Normalization / anchoring.
    normalize_ts: bool = False
    start_ts: int = 0
    first_ts: int = 0
    current_ts: int = 0
    anchored: bool = False

    # Stream position / progress.
    eof: bool = False
    n_read_events: int = 0

    # Windowing config (the reader's effective delta_t / n_events defaults).
    delta_t: int = 10_000
    n_events: int = 1_000_000
    read_external_triggers: bool = False

    # Staging accumulator (allocated lazily on first use).
    accumulator: "EventAccumulator | None" = None
    acc_capacity: int = 1 << 20

    # Native-fill fast path.
    native_fill: bool = False
    step: int = 1 << 20

    # delta_t fast-path state (see DeltaTStrategy).
    dt_step: int = 1 << 17
    dt_carry: "EventSoABuffers | None" = None
    dt_est: int = 1 << 20
    dt_tr: "TriggerSoABuffers | None" = None

    # delta_t buffer recycling (reuse_buffers ring).
    reuse_buffers: bool = False
    dt_slots: list = field(default_factory=list)
    dt_slot_i: int = 0

    # Ring-sizing inputs (mirror EventReader's async settings).
    async_read: bool = False
    prefetch_depth: "int | None" = None

    # ------------------------------------------------------------------ #
    # Accumulator helpers
    # ------------------------------------------------------------------ #
    def ensure_accumulator(self) -> "EventAccumulator":
        """Allocate the staging accumulator on first use and return it."""
        if self.accumulator is None:
            self.accumulator = EventAccumulator(self.acc_capacity)
        return self.accumulator

    # ------------------------------------------------------------------ #
    # Anchoring
    # ------------------------------------------------------------------ #
    def anchor_first_ts(self, ts: int) -> None:
        """Record ``ts`` as the stream's first timestamp / window origin.

        Sets ``first_ts``, resets the window clock to it, and marks the context
        anchored (mirrors the ``self._first_ts = ...; self._current_ts = ...;
        self._anchored = True`` triple the monolith repeated).
        """
        self.first_ts = int(ts)
        self.current_ts = int(ts)
        self.anchored = True

    # ------------------------------------------------------------------ #
    # delta_t fast-path helpers (ported from the monolith verbatim)
    # ------------------------------------------------------------------ #
    def dt_trigger_sink(self) -> "TriggerSoABuffers":
        """Lazily-allocated throwaway trigger sink for the delta_t fast path.

        Reset before every ``parse_step`` so it never fills and stalls the
        parser on a trigger-dense region. Only used when the caller did not
        request external triggers, so its contents are always discarded.
        """
        if self.dt_tr is None:
            self.dt_tr = TriggerSoABuffers(max(self.step // 16, 1024))
        return self.dt_tr

    @staticmethod
    def grow_out(out: "EventSoABuffers", step: int) -> None:
        """Ensure ``out`` has room for one more ``step`` events, then cap the SoA
        capacity the parser sees to ``size + step`` so a single ``parse_step``
        overshoots the time window by at most one step's worth."""
        if out.capacity - out.size < step:
            out.grow(max(out.size + step, int(out.capacity * 1.5) + 1))
        out.c.capacity = out.size + step

    def dt_ring_size(self) -> int:
        """Number of recycled window buffers needed so no live window is
        overwritten: the consumer's current window plus one being decoded
        (sync), plus everything an async prefetch may hold queued."""
        if self.async_read:
            from .._prefetch import DEFAULT_DEPTH
            depth = self.prefetch_depth if self.prefetch_depth is not None else DEFAULT_DEPTH
            return depth + 2
        return 2

    def acquire_window_buffer(self, need: int) -> "EventSoABuffers":
        """Decode buffer for one delta_t window, sized for ``need`` events.

        Default: a fresh buffer per window (independent result, safe to keep).
        With ``reuse_buffers``: cycle the persistent ring -- pages stay warm, no
        per-window allocation -- and the returned window aliases the slot until
        the ring wraps back to it.
        """
        if not self.reuse_buffers:
            return EventSoABuffers(need)
        slots = self.dt_slots
        i = self.dt_slot_i
        if i >= len(slots):
            slots.append(EventSoABuffers(need))
        self.dt_slot_i = (i + 1) % self.dt_ring_size()
        buf = slots[i]
        if buf.capacity < need:
            buf.grow(need)
        buf.c.size = 0
        buf.c.capacity = buf.capacity
        return buf

    def peek_first_ts(self) -> "int | None":
        """Timestamp of the stream's first event, without disturbing later reads.

        Decodes a small scratch batch to learn the first timestamp; the caller
        then :meth:`~decoder.reset`\\ s the decoder so the real window-0 decode
        starts cleanly from the beginning. Returns ``None`` if the stream is
        empty.
        """
        dec = self.decoder
        scratch = EventSoABuffers(4096)
        tr = self.dt_trigger_sink()
        while scratch.size == 0 and not dec.is_eof():
            tr.reset()
            dec.parse_step(scratch, tr)  # type: ignore[attr-defined]
        return int(scratch.t[0]) if scratch.size else None

    def flush_dt_carry(self) -> None:
        """Fold a pending delta_t fast-path overshoot into the accumulator.

        The fast paths keep up to one decode step of overshoot in ``dt_carry``;
        any code path that reads via the accumulator (mixed-mode overrides,
        ``read_all``) must fold it in first or those events are silently lost.
        Allocates the accumulator if needed.
        """
        carry = self.dt_carry
        if carry is None or carry.size == 0:
            self.dt_carry = None
            return
        acc = self.ensure_accumulator()
        acc.append(events_view(carry))
        self.dt_carry = None
        # Carried events exist => stream is not exhausted for the reader even
        # if the decoder itself hit EOF.
        self.eof = False
