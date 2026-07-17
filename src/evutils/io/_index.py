"""Seek index for timestamp / event-index random access.

A :class:`SeekIndex` is a coarse, monotonic map from *absolute timestamp* and
*cumulative event count* to a *word offset* into a decoder's payload. It lets a
seekable decoder jump close to a target (the nearest bookmark at or before it)
and then decode forward the small remainder to the exact position.

Two ways to obtain one:

* **Build it** from a fresh sequential decode pass (:func:`build_seek_index`),
  recording one bookmark per parse step. The timestamps recorded this way are
  the decoder's own absolute timestamps, so they already include any TIME_HIGH
  wrap accumulation -- which is exactly what a mid-file jump + parser reset
  would otherwise lose.
* **Read an OpenEB / Metavision** ``<file>.raw.tmp_index`` sidecar
  (:func:`read_metavision_index`), reusing an index another tool already built.
  Metavision timestamps are shifted by ``ts_shift_us`` relative to the raw
  stream; we add it back so bookmarks live in the same (raw-absolute) timeline
  the evutils EVT decoder produces.

The wrap accumulator that a fresh ``parser.reset()`` cannot recover is restored
at seek time from a bookmark's absolute timestamp: ``W = bookmark_ts -
first_decoded_ts`` (a multiple of the format's wrap period), added to every
timestamp decoded after the jump.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


from typing import Protocol, TYPE_CHECKING
if TYPE_CHECKING:
    import numpy as np

class SeekIndex(Protocol):
    """Monotonic (timestamp, cumulative-count) -> word-offset bookmarks."""

    @property
    def n_events(self) -> int | None:
        """Total event count of the indexed stream (None if not fully indexed)."""
        ...

    def bookmark_for_time(self, t: int) -> tuple[int, int, int]:
        """Return (word_offset, cum_count, ts) for the last bookmark whose timestamp is <= t."""
        ...

    def bookmark_for_event(self, n: int) -> tuple[int, int, int]:
        """Return (word_offset, cum_count, ts) for the last bookmark whose cumulative count is <= n."""
        ...

@dataclass
class StaticSeekIndex:
    """A fully built, read-only seek index."""
    ts: np.ndarray
    word_offset: np.ndarray
    cum_count: np.ndarray
    n_events: int

    def bookmark_for_time(self, t: int) -> tuple[int, int, int]:
        if len(self.ts) == 0:
            return 0, 0, 0
        i = int(np.searchsorted(self.ts, t, side="right")) - 1
        i = max(i, 0)
        return int(self.word_offset[i]), int(self.cum_count[i]), int(self.ts[i])

    def bookmark_for_event(self, n: int) -> tuple[int, int, int]:
        if len(self.cum_count) == 0:
            return 0, 0, 0
        i = int(np.searchsorted(self.cum_count, n, side="right")) - 1
        i = max(i, 0)
        return int(self.word_offset[i]), int(self.cum_count[i]), int(self.ts[i])

    def __len__(self) -> int:
        return len(self.ts)

class IncrementalSeekIndex:
    """A seek index that builds itself by decoding words on demand.

    Bookmarks are aligned to TIME_HIGH words: each segment boundary is the
    first TIME_HIGH at/after the previous boundary plus ``stride_words``. This
    is what makes the seek-time wrap correction sound -- a parser reset at a
    bookmark re-establishes its absolute time base *before* decoding any
    event, so ``bookmark_ts - first_decoded_ts`` is an exact multiple of the
    format's wrap period. A bookmark at an arbitrary parse-step boundary does
    NOT have that property (events decoded before the next TIME_HIGH carry
    only their low timestamp bits, and the snap lands a whole wrap period
    off).

    ``time_high`` is the ``(type-shift, type-code)`` descriptor of the
    format's TIME_HIGH word. Without it (no descriptor known) the index
    degrades to a single bookmark at ``start_offset`` -- correct, just no
    mid-file acceleration.
    """
    def __init__(self, words: np.ndarray, start_offset: int, input_cls: type, parser_cls: type,
                 tail_pad: int, word_dtype: np.dtype, chunk_cap: int = 65_536,
                 time_high: "tuple[int, int] | None" = None, stride_words: int = 1 << 16):
        self._words = words
        self._start_offset = start_offset
        self._input_cls = input_cls
        self._parser = parser_cls()
        self._tail_pad = tail_pad
        self._word_dtype = word_dtype
        self._time_high = time_high
        self._stride = max(int(stride_words), 64)

        cap = max(int(chunk_cap), 128)
        from ._native_core import EventSoABuffers, TriggerSoABuffers
        self._ev = EventSoABuffers(cap)
        self._tr = TriggerSoABuffers(max(cap // 16, 1))

        self._ts_list: list[int] = []
        self._off_list: list[int] = []
        self._cum_list: list[int] = []

        self._cum = 0
        self._off = int(start_offset)
        self._is_eof = False

        self._ts_arr = np.empty(0, dtype=np.int64)
        self._off_arr = np.empty(0, dtype=np.int64)
        self._cum_arr = np.empty(0, dtype=np.int64)
        self._arrs_stale = False

    @property
    def n_events(self) -> int | None:
        return self._cum if self._is_eof else None

    def _update_arrs(self):
        if self._arrs_stale:
            self._ts_arr = np.asarray(self._ts_list, dtype=np.int64)
            self._off_arr = np.asarray(self._off_list, dtype=np.int64)
            self._cum_arr = np.asarray(self._cum_list, dtype=np.int64)
            self._arrs_stale = False

    def _next_time_high(self, start: int) -> int:
        """Word offset of the first TIME_HIGH at/after ``start`` (or n_words).

        Vectorized block scan, same shape as the decoder's
        ``_find_first_time_high``.
        """
        n = len(self._words)
        if self._time_high is None:
            return n
        shift, code = self._time_high
        start = min(max(start, 0), n)
        block = 1 << 16
        while start < n:
            stop = min(start + block, n)
            seg = self._words[start:stop]
            hits = ((seg >> shift) & 0xF) == code
            i = int(np.argmax(hits))
            if hits[i]:
                return start + i
            start = stop
            block = min(block * 4, 1 << 24)
        return n

    def _build_until(self, target_t: int | None = None, target_n: int | None = None) -> None:
        if self._is_eof:
            return
        if target_t is not None and len(self._ts_list) > 0 and self._ts_list[-1] >= target_t:
            return
        if target_n is not None and self._cum >= target_n:
            return

        from ._native_core import events_view, parse_step
        words = self._words
        n_words = len(words)
        while self._off < n_words:
            seg_start = self._off
            seg_end = self._next_time_high(seg_start + self._stride)
            first_ts: int | None = None
            cum0 = self._cum

            # Decode the whole segment [seg_start, seg_end); parser state is
            # continuous across segments, only the *bookmark offsets* are
            # TIME_HIGH-aligned.
            while self._off < n_words and self._off < seg_end:
                final = seg_end >= n_words
                self._ev.reset()
                self._tr.reset()
                appended, new_off = parse_step(
                    words if final else words[:seg_end], self._off,
                    self._input_cls, self._parser, self._ev, self._tr,
                    tail_pad=self._tail_pad if final else 0,
                    word_dtype=self._word_dtype,
                )
                if appended == 0 and new_off <= self._off:
                    self._is_eof = True  # zero progress on full input
                    break
                if (not final and self._tail_pad and appended == 0
                        and new_off >= seg_end):
                    # Look-ahead formats (EVT3) stall a few words short of the
                    # sliced boundary and parse_step would skip that residue.
                    # Flush it through a zero-padded scratch copy: counts and
                    # timestamps stay exact (only decoder x/y state, which the
                    # index never records, can be perturbed by the pad words).
                    tail = words[self._off:seg_end]
                    if len(tail):
                        scratch = np.zeros(len(tail) + self._tail_pad,
                                           dtype=self._word_dtype)
                        scratch[:len(tail)] = tail
                        self._ev.reset()
                        self._tr.reset()
                        parse_step(scratch, 0, self._input_cls, self._parser,
                                   self._ev, self._tr, tail_pad=0,
                                   word_dtype=self._word_dtype)
                        appended = self._ev.size
                        if appended and first_ts is None:
                            first_ts = int(events_view(self._ev).t[0])
                        self._cum += appended
                    self._off = seg_end
                    break
                if appended:
                    if first_ts is None:
                        first_ts = int(events_view(self._ev).t[0])
                    self._cum += appended
                self._off = new_off

            if first_ts is not None:
                self._ts_list.append(first_ts)
                self._off_list.append(seg_start)
                self._cum_list.append(cum0)
                self._arrs_stale = True

            if self._is_eof or self._off >= n_words:
                self._is_eof = True
                break
            if target_t is not None and first_ts is not None and first_ts >= target_t:
                break
            if target_n is not None and self._cum >= target_n:
                break

    def bookmark_for_time(self, t: int) -> tuple[int, int, int]:
        self._build_until(target_t=t)
        self._update_arrs()
        if len(self._ts_arr) == 0:
            return self._start_offset, 0, 0
        i = int(np.searchsorted(self._ts_arr, t, side="right")) - 1
        i = max(i, 0)
        return int(self._off_arr[i]), int(self._cum_arr[i]), int(self._ts_arr[i])

    def bookmark_for_event(self, n: int) -> tuple[int, int, int]:
        self._build_until(target_n=n)
        self._update_arrs()
        if len(self._cum_arr) == 0:
            return self._start_offset, 0, 0
        i = int(np.searchsorted(self._cum_arr, n, side="right")) - 1
        i = max(i, 0)
        return int(self._off_arr[i]), int(self._cum_arr[i]), int(self._ts_arr[i])


# --------------------------------------------------------------------------- #
# Metavision / OpenEB `<file>.raw.tmp_index` sidecar
# --------------------------------------------------------------------------- #

#: On-disk bookmark record: (int64 timestamp, uint64 byte_offset, uint32 count).
#: Serialization order per OpenEB's serialize_bookmark (NOT struct order).
_MV_RECORD = np.dtype([("ts", "<i8"), ("byte_offset", "<u8"), ("count", "<u4")])
_MV_RECORD_SIZE = 20  # packed; np.dtype above is already 20 (no padding)


def metavision_index_path(raw_path: "str | Path") -> Path:
    """Sidecar path for a raw file: ``<file>.raw.tmp_index``."""
    p = Path(raw_path)
    return p.with_name(p.name + ".tmp_index")


def _parse_mv_header(buf: bytes) -> tuple[dict[str, str], int]:
    """Parse the ``% key value`` text header; return (fields, payload offset)."""
    fields: dict[str, str] = {}
    off = 0
    n = len(buf)
    while off < n and buf[off:off + 1] == b"%":
        nl = buf.find(b"\n", off)
        if nl < 0:
            break
        line = buf[off:nl].decode("ascii", "ignore").strip()
        off = nl + 1
        if line == "% end":
            break
        parts = line.split(None, 2)  # "%", key, value
        if len(parts) >= 3:
            fields[parts[1].lower()] = parts[2]
    return fields, off


def read_metavision_index(index_path: "str | Path", raw_path: "str | Path",
                          payload_off: int, word_size: int) -> "SeekIndex | None":
    """Read a Metavision ``.tmp_index`` sidecar into a :class:`SeekIndex`.

    Returns ``None`` if the sidecar is missing or stale (its stored ``size``
    does not match the raw file's current byte size, matching Metavision's
    freshness check). Byte offsets are converted to payload word offsets and
    timestamps are shifted by ``ts_shift_us`` into the raw-stream timeline.
    """
    index_path = Path(index_path)
    raw_path = Path(raw_path)
    if not index_path.is_file() or not raw_path.is_file():
        return None

    data = index_path.read_bytes()
    fields, body = _parse_mv_header(data)

    # Freshness: stored size must equal the current raw byte size.
    try:
        if int(fields.get("size", "-1")) != raw_path.stat().st_size:
            return None
    except ValueError:
        return None

    ts_shift = 0
    try:
        ts_shift = int(fields.get("ts_shift_us", "0"))
    except ValueError:
        ts_shift = 0

    payload = data[body:]
    n_rec = len(payload) // _MV_RECORD_SIZE
    if n_rec <= 1:
        return None
    recs = np.frombuffer(payload, dtype=_MV_RECORD, count=n_rec)

    # The final record is the magic-number completeness marker (random bytes),
    # not a bookmark -- drop it. Then drop leading pre-time-base bookmarks
    # (ts < 0) and shift the rest into the raw-absolute timeline.
    recs = recs[:-1]
    recs = recs[recs["ts"] >= 0]
    if len(recs) == 0:
        return None

    ts = recs["ts"].astype(np.int64) + ts_shift
    byte_off = recs["byte_offset"].astype(np.int64)
    word_offset = (byte_off - int(payload_off)) // int(word_size)
    cum_count = np.cumsum(recs["count"].astype(np.int64)) - recs["count"].astype(np.int64)

    return StaticSeekIndex(
        ts=ts,
        word_offset=word_offset,
        cum_count=cum_count,
        n_events=int(cum_count[-1] + recs["count"][-1]),
    )
