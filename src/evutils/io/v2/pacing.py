"""Real-time playback pacing for the decomposed :class:`EventReader` (V2).

``Pacer`` reproduces the monolith's ``_pace`` / ``_paced_iter`` behaviour: each
chunk is released only once wall-clock time (anchored at the first delivered
chunk) reaches the event time it covers, so the stream plays back as if live.
``max_gap`` dead-air skipping is preserved (pinned by ``test_playback.py``).
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any, Iterator

if TYPE_CHECKING:
    from ...types import EventArray


class Pacer:
    """Paces chunk delivery to the recording's own timeline.

    Parameters
    ----------
    playback_speed : float
        Playback rate multiplier (``2.0`` = twice as fast as the recording).
    max_gap : float or None
        Longest idle stretch, in real seconds, the pacer will sleep through
        before skipping the dead air. ``None`` disables the skip (strict real
        time).
    """

    def __init__(self, playback_speed: float = 1.0, max_gap: "float | None" = 1.0):
        self.playback_speed = float(playback_speed)
        self.max_gap = float(max_gap) if max_gap is not None else None
        self._anchor: "tuple[float, int] | None" = None  # (wall time, event ts)

    def reset(self) -> None:
        """Drop the wall-clock anchor (re-anchors on the next paced chunk)."""
        self._anchor = None

    def pace(self, out: "Any") -> None:
        """Sleep until ``out``'s last event timestamp aligns with wall-clock
        playback time (anchored at the first delivered chunk).

        The anchor is absolute, so time the caller spends between chunks is
        subtracted from the delay automatically; when the target time is already
        past (slow decode or slow consumer), no delay is added.
        """
        ev = out[0] if isinstance(out, tuple) else out
        if len(ev) == 0:
            return
        t_last = int(ev.t[-1])
        now = time.perf_counter()
        if self._anchor is None:
            self._anchor = (now, int(ev.t[0]))
        wall0, ts0 = self._anchor
        target = wall0 + (t_last - ts0) / (1e6 * self.playback_speed)
        delay = target - now
        # A wait longer than max_gap means the recording went idle: skip the dead
        # air by re-anchoring to now, rather than stalling on one long sleep.
        if self.max_gap is not None and delay > self.max_gap:
            self._anchor = (now, t_last)
            return
        if delay > 0:
            time.sleep(delay)

    def paced_iter(self, it: "Iterator[Any]") -> "Iterator[Any]":
        """Wrap an iterator so each chunk is released on the playback clock."""
        try:
            for chunk in it:
                self.pace(chunk)
                yield chunk
        finally:
            # Propagate early termination (break / close) to a prefetching
            # iterator so its worker thread is released.
            if hasattr(it, "close"):
                it.close()
