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


@dataclass
class SeekIndex:
    """Monotonic (timestamp, cumulative-count) -> word-offset bookmarks.

    Attributes
    ----------
    ts : np.ndarray[int64]
        Absolute timestamp (µs, raw-stream timeline) of the first event at each
        bookmark's ``word_offset``.
    word_offset : np.ndarray[int64]
        Word offset into the decoder payload where decoding may resume (a point
        from which the decoder re-establishes an absolute time base).
    cum_count : np.ndarray[int64]
        Number of events before each bookmark (cumulative from the start).
    n_events : int
        Total event count of the indexed stream.
    """

    ts: np.ndarray
    word_offset: np.ndarray
    cum_count: np.ndarray
    n_events: int

    def bookmark_for_time(self, t: int) -> int:
        """Index of the last bookmark whose timestamp is ``<= t`` (>= 0)."""
        i = int(np.searchsorted(self.ts, t, side="right")) - 1
        return max(i, 0)

    def bookmark_for_event(self, n: int) -> int:
        """Index of the last bookmark whose cumulative count is ``<= n`` (>= 0)."""
        i = int(np.searchsorted(self.cum_count, n, side="right")) - 1
        return max(i, 0)

    def __len__(self) -> int:
        return len(self.ts)


def build_seek_index(*, words, start_offset, input_cls, parser_cls,
                     tail_pad, word_dtype, chunk_cap: int = 65_536) -> SeekIndex:
    """Build a :class:`SeekIndex` by decoding ``words`` sequentially.

    Records one bookmark per parse step: the word offset it started at, the
    absolute timestamp of that step's first event, and the cumulative count so
    far. ``chunk_cap`` bounds events per step, hence the forward-decode work
    after a seek lands on a bookmark.
    """
    from ._native_core import (
        EventSoABuffers,
        TriggerSoABuffers,
        events_view,
        parse_step,
    )

    parser = parser_cls()
    cap = max(int(chunk_cap), 128)
    ev = EventSoABuffers(cap)
    tr = TriggerSoABuffers(max(cap // 16, 1))

    ts_list: list[int] = []
    off_list: list[int] = []
    cum_list: list[int] = []
    cum = 0
    off = int(start_offset)
    n = len(words)

    while off < n:
        boff = off
        ev.reset()
        tr.reset()
        appended, off = parse_step(
            words, off, input_cls, parser, ev, tr,
            tail_pad=tail_pad, word_dtype=word_dtype,
        )
        if appended == 0:
            if off <= boff:
                break  # no progress -> only sub-padding tail remains
            continue
        ts_list.append(int(events_view(ev).t[0]))
        off_list.append(boff)
        cum_list.append(cum)
        cum += appended

    return SeekIndex(
        ts=np.asarray(ts_list, dtype=np.int64),
        word_offset=np.asarray(off_list, dtype=np.int64),
        cum_count=np.asarray(cum_list, dtype=np.int64),
        n_events=cum,
    )


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

    return SeekIndex(
        ts=ts,
        word_offset=word_offset,
        cum_count=cum_count,
        n_events=int(cum_count[-1] + recs["count"][-1]),
    )
