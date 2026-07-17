"""Event reader module.

Provides the `EventReader` class for reading and decoding event data
from a file or byte stream.
"""

import io
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

import numpy as np

from ..types import EventArray, TriggerArray

_EMPTY_EVENTS = EventArray.empty()
from . import decoders as ev_decoders
from ._native_core import (EVUTILS_PARSE_OUTPUT_FULL,
                           EVUTILS_PARSE_WINDOW_DONE, EventSoABuffers,
                           TriggerSoABuffers, events_view)
from ._prefetch import PrefetchIterator
from ._source import ByteSource, make_source
from .buffer import EventAccumulator


class EventReader():
    """Class for reading and automatically decoding and slicing events from a file or stream.

    The reader supports different modes of operation, including reading a fixed number of events, reading events within a time window, or a combination of both.
    The file_decoder is chosen automatically based on the file format or can be supplied explicitly.

    Parameters
    ----------
    file: Path or str or io.BufferedReader or bytes or ByteSource
        Path to the data file or a readable stream or bytes or a ByteSource
    delta_t: int or optional
        Time window in microseconds, by default None
    n_events: int or None
        Number of events to read in a chunk, by default None
    max_events: int, default=10_000_000
        Maximum number of events to read at once
    mode: {'delta_t', 'n_events', 'mixed', 'all', 'auto'}, default 'auto'
        Mode of operation ```["delta_t", "n_events", "mixed", "all", "auto"]```
    start_ts: int, default=0
        Start timestamp offset for the events, by default 0 (start of the file)
    normalize_ts: bool, default=False
        Normalize timestamps to start from zero
    max_time: int, default=1_000_000_000_000
        Maximum timestamp to read
    width: int or None
        Width of the frame, by default inferred from the file
    height: int or None
        Height of the frame, by default inferred from the file
    ext_trigger: bool, default=False
        Whether to read external trigger events, by default False
    async_read: bool, default=False
        Decode ahead in a background thread while iterating: the next window
        is parsed while the caller processes the current one (the native
        parsers release the GIL). Helps whenever per-chunk processing takes
        meaningful time (numpy pipelines, GPU inference -- the read becomes
        essentially free); can slightly hurt when the processing already
        saturates memory bandwidth. Affects iteration only; ``read()`` /
        ``read_all()`` stay synchronous and raise while an asynchronous
        iterator is active.
    prefetch_depth: int or None, default=None
        How many decoded windows the ``async_read`` worker may buffer ahead of
        the consumer (default 2 when ``None``). Deeper queues cost memory
        (depth x window size) but ride out I/O stalls: for real-time playback
        from a cold file, a depth of ~6 absorbs multi-window disk page-fault
        hitches that a depth of 2 cannot. Ignored without ``async_read``.
    reuse_buffers: bool, default=False
        Recycle the decode buffers of ``delta_t`` windows (openeb-style ring)
        instead of allocating a fresh buffer per window. Large windows make the
        per-window allocation the dominant cost (page-faulting hundreds of MB
        every window); with recycling the buffer pages stay warm and delta_t
        reads run at ~the raw decoder ceiling. The returned array then
        **aliases** a recycled buffer: it is valid until ``prefetch_depth``
        further windows have been read (or one window, without ``async_read``)
        -- copy it if you keep it longer, exactly like ``EventStreamer``
        chunks. Off by default: every window is an independent array.
    real_time: bool, default=False
        Pace :meth:`read` and iteration to the recording's own timeline: each
        chunk is released only once the wall-clock time since the first chunk
        matches the event time it covers, so the stream plays back as if live.
        Pacing is anchored to an absolute start time, so time spent processing
        a chunk is absorbed automatically (no drift); if decoding or
        processing falls behind, chunks are released immediately with no added
        delay. ``read_all()`` is never paced. Combine with ``async_read=True``
        to decode ahead while waiting out the pacing delay.
    playback_speed: float, default=1.0
        Playback rate multiplier for ``real_time`` mode: ``2.0`` plays twice
        as fast as the recording, ``0.5`` at half speed. Ignored when
        ``real_time=False``.
    max_gap: float or None, default=1.0
        Longest idle stretch, in real seconds, that ``real_time`` pacing will
        sleep through. If a recording goes silent for longer (e.g. a
        late-starting stream, or a pause mid-recording), the pacer skips the
        dead air and resumes immediately instead of blocking on one long
        sleep. ``None`` disables the skip (strict real time). Ignored when
        ``real_time=False``.
    index: str or bool, default="auto"
        Seek-index policy for random access (:meth:`seek`). ``"auto"``/``True``/
        ``False`` all build an exact in-memory index lazily on the first seek
        (``False`` additionally never touches a sidecar). ``"metavision"`` reads
        a Metavision ``.tmp_index`` sidecar instead -- fast, but approximate near
        large event gaps. Only EVT formats use an index; others seek by record
        math or ``searchsorted``.
    strict: bool, default=False
        Corrupt-packet policy. When False (default), a malformed packet is
        skipped with a :class:`UserWarning` and decoding resumes -- a robust
        decoder (like Metavision's UNRELIABLE mode). When True, a malformed
        packet raises instead (a SAFE decoder). Warnings can be captured with
        :func:`warnings.catch_warnings`.
    file_decoder: ev_decoders.EventDecoder or type[ev_decoders.EventDecoder] or None, default=None
        File decoder to use, by default None - automatic
    **kwargs
        Additional arguments to pass to the file decoder

    Raises
    ------
    ValueError
        If the mode is not supported or if the delta_t or n_events are not specified when needed

    Examples
    --------
    >>> from evutils.io import EventReader
    >>> # Read events in chunks of 10ms (10,000 microseconds)
    >>> reader = EventReader(b"% format EVT3;height=720;width=1280\\n% end\\n", delta_t=10000)
    >>> for events in reader:
    ...     # Access fields directly
    ...     x, y, p, t = events.x, events.y, events.p, events.t
    ...     print(f"Processed {len(events)} events")
    Processed 0 events

    TODO:
    Architectural Refactor Roadmap:
    Decouple `EventReader`'s monolithic design by extracting the core byte-to-array decoding into
    a native `EventStreamer`. The slicing logic (`mode="delta_t"`, `mode="n_events"`, etc.) should
    be moved to composable pipeline generators in `evutils.chunking`. `EventReader` will then
    become a simple Façade that dynamically assembles these pipeline generators to maintain
    this exact backward-compatible API, while allowing power-users to compose custom pipelines
    (e.g., slicing by external trigger boundaries).
    """

    READING_MODES = ["delta_t", "n_events", "mixed", "all", "auto"]
    DEFAULT_N_EVENTS = 1_000_000
    DEFAULT_DELTA_T = 10_000
    def __init__(self, file: Path | str | io.BufferedReader | bytes | ByteSource,
                 delta_t:int|None=None,
                 n_events:int|None=None,
                 mode:str="auto",
                 start_ts:int=0,
                 normalize_ts: bool=False,
                 max_time:int=1_000_000_000_000,
                 max_events:int=10_000_000,
                 width:int | None=None, height:int | None=None,
                 ext_trigger: bool=False,
                 async_read: bool=False,
                 prefetch_depth: int | None=None,
                 reuse_buffers: bool=False,
                 real_time: bool=False,
                 playback_speed: float=1.0,
                 max_gap: float | None=1.0,
                 index: "str | bool" = "auto",
                 strict: bool=False,
                 batch_mode: bool=False,
                 file_decoder: ev_decoders.EventDecoder | type[ev_decoders.EventDecoder] | None = None,
                 **kwargs) -> None:

        # Remember the path (if any) for repr / reset semantics.
        self._file_name: Path | None = Path(file) if isinstance(file, (str, Path)) else None
        self._read_external_triggers = ext_trigger
        self._batch_mode = batch_mode

        # 1. Normalise the input into a ByteSource (path | stream | bytes |
        #    BytesIO | ByteSource -> ByteSource). Regular files are memory-mapped.
        self._source: ByteSource = make_source(file)

        # 2. Resolve the decoder and launch it:
        #    explicit instance > explicit class > heuristic (extension, then
        #    content sniffing of the source).
        if isinstance(file_decoder, ev_decoders.EventDecoder):
            self._file_decoder = file_decoder
        else:
            decoder_cls = file_decoder or ev_decoders.resolve_decoder_cls(self._source)
            self._file_decoder = decoder_cls(self._source, **kwargs)

        self._file_decoder.read_external_triggers = self._read_external_triggers
        # Corrupt-packet policy: strict=True re-raises the decoder's WARNING
        # (a malformed packet) instead of skipping it. Default is robust
        # (warn + skip + resume), like Metavision's UNRELIABLE decoder.
        self._file_decoder._strict = strict
        if self._read_external_triggers and not self._file_decoder.SUPPORTS_EXT_TRIGGERS:
            import warnings
            warnings.warn(f"{self._file_decoder.__class__.__name__} does not support reading external triggers.")

        # This will now be io.BufferedReader
        self._eof = False

        # If not defined explicitly, the width and height are fetch from the file (not all formats support this)
        self._width = width
        self._height = height
        self._start_ts = start_ts # Normalization origin: with normalize_ts=True the earliest event's timestamp becomes start_ts. NOT a seek -- use seek() to skip to a time.
        self._first_ts = 0 # First timestamp in the file, used for normalization
        self._current_ts = self._first_ts
        self._normalize_ts = normalize_ts # Normalize timestamps to start from zero

        # Validate the parameters
        if mode not in EventReader.READING_MODES:
            raise ValueError(f"Mode {mode} not supported. Supported modes are: {EventReader.READING_MODES}")

        self._mode = mode.lower()

        # if mode is auto, we will try to infer the mode from the parameters
        if self._mode == "auto":
            # If both delta_t and n_events are specified, we will use mixed mode
            if delta_t is not None and n_events is not None:
                self._mode = "mixed"

            # If only one of the parameters is specified, we will use that mode, the other will be set to the maximum
            elif delta_t is not None:
                self._mode = "delta_t"
                n_events = max_events
            elif n_events is not None:
                self._mode = "n_events"
                delta_t = max_time
            else:
                # If none of the parameters are specified, we will use the default Values
                self._mode = "mixed"
                delta_t = self.DEFAULT_DELTA_T
                n_events = self.DEFAULT_N_EVENTS

        # If the mode is not auto, we will check if the parameters are specified
        elif self._mode == "delta_t":
            if delta_t is None:
                raise ValueError("delta_t must be specified")
            if n_events is None:
                n_events = max_events
        elif self._mode == "n_events":
            if n_events is None:
                raise ValueError("n_events must be specified")
            if delta_t is None:
                delta_t = max_time
        elif self._mode == "mixed":
            if delta_t is None:
                delta_t = self.DEFAULT_DELTA_T
            if n_events is None:
                n_events = self.DEFAULT_N_EVENTS

        elif self._mode == "all":
            delta_t = max_time
            n_events = max_events

        # Validate the parameters
        if delta_t is None:
            delta_t = self.DEFAULT_DELTA_T
        if n_events is None:
            n_events = self.DEFAULT_N_EVENTS

        if not isinstance(delta_t, int):
            raise TypeError("delta_t must be an integer")

        if not isinstance(n_events, int):
            raise TypeError("n_events must be an integer")

        if delta_t <= 0:
            raise ValueError("delta_t must be positive")

        if n_events <= 0:
            raise ValueError("n_events must be positive")

        # delta_t and n_events to read on each call
        self._delta_t = delta_t
        self._n_events = n_events

        # Maximum number of events to read and maximum time to read in a chunk
        self._max_events = max_events if max_events < self._n_events else self._n_events
        self._max_time = max_time if max_time < self._delta_t else self._delta_t

        self._is_initialized = False

        # Windowed reads decode straight into this reused accumulator (no
        # intermediate copy); only the returned window is copied out. Allocated
        # lazily on first read() so the read_all() fast path never pays for it.
        # Granularity of a single decode step, and a native fast path flag.
        self._buffer: EventAccumulator | None = None
        self._step = 1 << 20
        # Staging capacity: one window plus a decode step of overshoot. Kept
        # deliberately small (and capped at a few steps) so the parser writes
        # into a cache/TLB-warm working set -- sizing it to the full max_events
        # up front inflated the buffer ~4x for the common 1M-event window and
        # roughly halved decode throughput. It grows on demand (EventAccumulator)
        # if an actual window -- or drain-to-EOF mode -- needs more, so this only
        # sets the starting point.
        self._acc_capacity = min(self._n_events, 4 * self._step) + 2 * self._step
        self._native_fill = hasattr(self._file_decoder, "parse_step")

        # delta_t fast-path state (see _read_delta_t_fast): the overshoot carried
        # from the previous window (events past its time boundary), a rolling
        # window-size estimate to size the next buffer, and a throwaway trigger
        # sink. Only used in pure delta_t mode with a native-fill decoder.
        self._dt_carry: EventSoABuffers | None = None
        self._dt_est = self._step
        self._dt_tr: TriggerSoABuffers | None = None
        # The fast path decodes in smaller steps than the accumulator so the
        # window's trailing overshoot (the only part copied) stays small; too
        # large and the carry copy rivals the full-window copy it replaces.
        self._dt_step = 1 << 17
        # Buffer recycling for delta_t windows (reuse_buffers=True): a small
        # ring of persistent decode buffers, cycled per window so their pages
        # stay warm and no per-window allocation/page-fault cost enters the hot
        # path. Ring size covers every window that can be alive at once: the
        # consumer's current one plus everything buffered by an async prefetch.
        self._reuse_buffers = bool(reuse_buffers)
        self._dt_slots: list[EventSoABuffers] = []
        self._dt_slot_i = 0

        # Asynchronous iteration: when enabled, __iter__ decodes ahead in a
        # worker thread (see io/_prefetch.py). The active iterator owns the
        # decoder until it is exhausted or closed.
        self._async_read = async_read
        if prefetch_depth is not None and prefetch_depth < 1:
            raise ValueError("prefetch_depth must be >= 1")
        self._prefetch_depth = prefetch_depth
        self._active_prefetch: PrefetchIterator | None = None

        # Real-time playback: chunks are released against an absolute
        # wall-clock anchor set when the first chunk is delivered, so the
        # stream is paced like a live sensor (see _pace()).
        if not isinstance(playback_speed, (int, float)) or playback_speed <= 0:
            raise ValueError("playback_speed must be a positive number")
        self._real_time = real_time
        self._playback_speed = float(playback_speed)
        # Longest idle stretch (real seconds) the pacer will sleep through. If a
        # window would require a longer wait -- a silent gap in the recording,
        # e.g. a late-starting stream -- the pacer skips the dead air and
        # re-anchors so playback resumes immediately. None = strict real time.
        if max_gap is not None and (not isinstance(max_gap, (int, float)) or max_gap <= 0):
            raise ValueError("max_gap must be a positive number or None")
        self._max_gap = float(max_gap) if max_gap is not None else None
        self._pace_anchor: tuple[float, int] | None = None  # (wall time, event ts)

        self._n_read_events = 0 # Number of events read (not includeing events stored in buffer)

        # Seeking (see seek()). ``_anchored`` decouples the "first timestamp"
        # anchoring from ``_n_read_events == 0`` so a post-seek read keeps the
        # sought ``_current_ts`` instead of re-anchoring to the stream start.
        self._anchored = False
        self._index_opt = index
        # ``index`` selects the seek index source for formats that use one (EVT):
        #   "auto"/True/False -> build an exact index in memory, lazily on the
        #                        first seek() (never on open);
        #   "metavision"      -> read a Metavision `.tmp_index` sidecar when
        #                        present (fast, but approximate near large event
        #                        gaps), else fall back to the built index.
        self._file_decoder._use_sidecar = (index == "metavision")
        if self._file_name is not None:
            self._file_decoder._raw_path = str(self._file_name)

    def init(self) -> None:
        """Initialize the reader, can be used explicitly or implicitly by the read method."""
        if self._is_initialized:
            return
        self._file_decoder.init()
        self._is_initialized = True

    def _check_no_active_prefetch(self) -> None:
        """Direct reads are not allowed while a prefetching iterator owns the
        decoder (its worker thread is reading concurrently)."""
        if self._active_prefetch is not None:
            raise RuntimeError(
                "An asynchronous iterator is active on this reader: exhaust or "
                "close() it before calling read()/read_all(), or iterate instead."
            )

    def _pull(self, acc: EventAccumulator, delta_t: int, n_events: int) -> int:
        """Pull more events into the accumulator, returning the number added
        (0 => end of stream). Native decoders decode straight into the
        accumulator's storage (no copy); others have their ``read_chunk`` output
        appended.

        Parameters
        ----------
        acc : EventAccumulator
            The accumulator to pull events into.
        delta_t : int
            The time window constraint for reading.
        n_events : int
            The number of events constraint for reading.

        Returns
        -------
        int
            The number of events added to the accumulator.

        """
        dec = self._file_decoder
        if self._native_fill:
            while True:
                if dec.is_eof():
                    return 0
                ev, tr = acc.prepare(self._step)
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

    def read(self, delta_t:int|None=None, n_events:int|None=None) -> 'EventArray | tuple[EventArray, TriggerArray]':
        """Read events on the files based on the mode and the parameters.

        Parameters
        ----------
        delta_t
            Override the delta_t parameter, otherwise the default value is used from the constructor
        n_events
            Override the n_events parameter, otherwise the default value is used from the constructor

        Returns
        -------
        EventArray
            An array with the events

        Examples
        --------
        >>> reader = EventReader(b"% format EVT3;height=720;width=1280\\n% end\\n", delta_t=10000)
        >>> events = reader.read()
        >>> print(len(events))
        0
        >>> # Override the time window for a single read
        >>> more_events = reader.read(delta_t=5000)

        """
        self._check_no_active_prefetch()
        out = self._read(delta_t, n_events)
        if self._real_time:
            self._pace(out)
        if self._batch_mode:
            from ..types import DataBatch, TriggerArray
            if isinstance(out, tuple):
                return DataBatch(events=out[0], triggers=out[1])
            else:
                return DataBatch(events=out, triggers=TriggerArray.empty())
        return out

    def _pace(self, out: "np.ndarray | EventArray") -> None:
        """Sleep until the chunk's last event timestamp aligns with wall-clock
        playback time (anchored at the first delivered chunk).

        The anchor is absolute, so time the caller spends between chunks is
        subtracted from the delay automatically; when the target time is
        already past (slow decode or slow consumer), no delay is added.
        """
        ev = out[0] if isinstance(out, tuple) else out
        if len(ev) == 0:
            return
        t_last = int(ev.t[-1])
        now = time.perf_counter()
        if self._pace_anchor is None:
            self._pace_anchor = (now, int(ev.t[0]))
        wall0, ts0 = self._pace_anchor
        target = wall0 + (t_last - ts0) / (1e6 * self._playback_speed)
        delay = target - now
        # A wait longer than max_gap means the recording went idle (e.g. a
        # silent lead-in before the action starts). Rather than stall on a
        # single multi-second sleep, skip the dead air: re-anchor to now so the
        # next window is paced relative to this one.
        if self._max_gap is not None and delay > self._max_gap:
            self._pace_anchor = (now, t_last)
            return
        if delay > 0:
            time.sleep(delay)

    def _read_n_events_fast(self, n_events: int) -> "np.ndarray | EventArray":
        """Decode ~``n_events`` straight into a fresh output buffer and hand it
        out with no staging copy.

        Valid only in pure ``n_events`` mode: the count cutoff is enforced by the
        output buffer's capacity (the parser stops when it is full), so there is
        no time boundary to ``searchsorted`` and no overshoot to carry in a
        staging accumulator. This turns the windowed read from two passes over
        the data (decode into the reused accumulator, then copy the window out)
        into one, roughly matching :class:`EventStreamer` while still returning
        an independent array (the buffer is fresh per call, never reused).
        """
        dec = self._file_decoder
        # Fresh per-call output; handed to the caller as zero-copy views, so it
        # must not be reused. This path is only taken for "exact-capacity"
        # decoders (one event per record: EVT2/EVT4/DAT), so the parser fills
        # to *exactly* n_events and stops -- a single parse_step completes the
        # window with no overshoot and no remainder to carry. (Vector formats
        # would stop a few short and a second parse_step on the full buffer is
        # misread as EOF, so they stay on the accumulator path.)
        out = EventSoABuffers(n_events)
        out.c.capacity = n_events
        # Throwaway trigger sink (this path is only taken when the caller did not
        # request triggers); reset each step so it never fills and stalls the
        # parser on a trigger-dense region.
        tr = TriggerSoABuffers(max(n_events // 16, 1024))
        while out.size < n_events and not dec.is_eof():
            tr.reset()
            before = out.size
            dec.parse_step(out, tr)
            if out.size == before and dec.is_eof():
                break
        self._eof = dec.is_eof()

        if out.size == 0:
            return EventArray.empty()
        if not self._anchored:
            self._first_ts = int(out.t[0])
            self._current_ts = self._first_ts
            self._anchored = True
        self._n_read_events += out.size

        output = events_view(out)
        if self._normalize_ts:
            output.t -= self._first_ts - self._start_ts
        return output

    def _read_n_events_readchunk(self, n_events: int) -> "np.ndarray | EventArray":
        """Hand out one ``read_chunk(n_events)`` result directly, no accumulator.

        For decoders whose ``read_chunk`` already returns an *independent* array
        of at most ``n_events`` events (``_independent_windows``): NPZ slices its
        loaded columns into fresh arrays, CSV parses into fresh arrays. Skipping
        the staging accumulator drops the append + slice_copy double copy.
        """
        dec = self._file_decoder
        chunk = dec.read_chunk(n_events_hint=n_events)
        if isinstance(chunk, tuple):
            chunk = chunk[0]
        # A full window is exactly n_events; anything short means the stream is
        # drained. (Do not use dec.is_eof(): for buffered decoders like CSV it
        # signals "input exhausted" while read_chunk still has buffered events.)
        self._eof = len(chunk) < n_events
        if len(chunk) == 0:
            return EventArray.empty()
        if self._n_read_events == 0:
            self._first_ts = int(chunk.t[0])
            self._current_ts = self._first_ts
        self._n_read_events += len(chunk)
        if self._normalize_ts:
            chunk.t -= self._first_ts - self._start_ts  # chunk is independent
        return chunk

    def _dt_trigger_sink(self) -> TriggerSoABuffers:
        """Lazily-allocated throwaway trigger sink for the delta_t fast path.

        Reset before every ``parse_step`` so it never fills and stalls the parser
        on a trigger-dense region. Only used when the caller did not request
        external triggers, so its contents are always discarded.
        """
        if self._dt_tr is None:
            self._dt_tr = TriggerSoABuffers(max(self._step // 16, 1024))
        return self._dt_tr

    @staticmethod
    def _grow_out(out: EventSoABuffers, step: int) -> None:
        """Ensure ``out`` has room for one more ``step`` events, then cap the SoA
        capacity the parser sees to ``size + step`` so a single ``parse_step``
        overshoots the time window by at most one step's worth."""
        if out.capacity - out.size < step:
            out.grow(max(out.size + step, int(out.capacity * 1.5) + 1))
        out.c.capacity = out.size + step

    def _read_delta_t_fast(self, delta_t: int) -> "np.ndarray | EventArray":
        """One ``delta_t`` window, fast path: decode into a fresh buffer, hand out
        a zero-copy view of the window, and carry only the small overshoot
        (events past the boundary, at most one decode step) to the next call.

        Mirrors :meth:`_read_n_events_fast` for time-windowed reads. The staging
        accumulator path copies the *entire* window out on every call
        (:meth:`EventAccumulator.slice_copy`) -- millions of events per window on
        a dense recording. Here the window is a view into a per-call buffer (never
        reused, so the view stays valid), and only the overshoot is copied.

        Valid only for native-fill decoders in pure ``delta_t`` mode with no
        external triggers (the guard in :meth:`_read` enforces this). The
        ``n_events`` safety cap (``max_events`` in delta_t mode) is honoured: a
        window larger than the cap is cut at the count boundary, exactly like the
        accumulator path.
        """
        dec = self._file_decoder
        step = self._dt_step
        n_cap = self._n_events
        tr = self._dt_trigger_sink()

        # Decode buffer (pooled or fresh), pre-sized to ~one window plus 25%
        # headroom so a normal window never triggers a mid-decode grow (a grow
        # reallocates + copies the whole buffer -- a large per-frame hitch).
        # Seed it with the previous call's overshoot.
        carry = self._dt_carry
        k = carry.size if carry is not None else 0
        need = max(self._dt_est * 5 // 4 + step, k + step)
        out = self._acquire_window_buffer(need)
        if k:
            out.t[:k] = carry.t[:k]
            out.x[:k] = carry.x[:k]
            out.y[:k] = carry.y[:k]
            out.p[:k] = carry.p[:k]
            out.c.size = k
        self._dt_carry = None

        # Anchor the stream's first timestamp / window origin on the very first
        # decoded event (only reachable on the first call, when there is no carry).
        if not self._anchored and out.size == 0:
            while out.size == 0 and not dec.is_eof():
                self._grow_out(out, step)
                tr.reset()
                dec.parse_step(out, tr)
            if out.size == 0:
                self._eof = True
                return EventArray.empty()
            self._first_ts = int(out.t[0])
            self._current_ts = self._first_ts
            self._anchored = True

        end_ts = self._current_ts + delta_t

        # Decode until the buffer holds an event at/after the time boundary, the
        # count cap is exceeded, or the stream ends.
        while not dec.is_eof():
            n = out.size
            if n and (int(out.t[n - 1]) >= end_ts or n > n_cap):
                break
            self._grow_out(out, step)
            tr.reset()
            added = dec.parse_step(out, tr)
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
            self._dt_carry = c

        out.c.size = idx
        # Track the largest window seen so the next buffer is pre-sized for it.
        self._dt_est = max(self._dt_est, idx)
        if count_cut:
            self._current_ts = int(t[idx])  # count cutoff: resume mid-window
        else:
            self._current_ts += delta_t
        self._n_read_events += idx
        # EOF only once the stream is drained *and* no overshoot remains, else the
        # final carried events would never be emitted (is_eof() stops iteration).
        self._eof = dec.is_eof() and self._dt_carry is None

        output = events_view(out)
        if self._normalize_ts:
            output.t -= self._first_ts - self._start_ts
        return output

    def _peek_first_ts(self) -> int | None:
        """Timestamp of the stream's first event, without disturbing later reads.

        Decodes a small scratch batch to learn the first timestamp; the caller
        then :meth:`~decoder.reset`\\ s the decoder so the real window-0 decode
        starts cleanly from the beginning. Returns ``None`` if the stream is
        empty.
        """
        dec = self._file_decoder
        scratch = EventSoABuffers(4096)
        tr = self._dt_trigger_sink()
        while scratch.size == 0 and not dec.is_eof():
            tr.reset()
            dec.parse_step(scratch, tr)
        return int(scratch.t[0]) if scratch.size else None

    def _dt_ring_size(self) -> int:
        """Number of recycled window buffers needed so no live window is
        overwritten: the consumer's current window plus one being decoded
        (sync), plus everything an async prefetch may hold queued."""
        if self._async_read:
            from ._prefetch import DEFAULT_DEPTH
            depth = self._prefetch_depth if self._prefetch_depth is not None else DEFAULT_DEPTH
            return depth + 2
        return 2

    def _acquire_window_buffer(self, need: int) -> EventSoABuffers:
        """Decode buffer for one delta_t window, sized for ``need`` events.

        Default: a fresh buffer per window (independent result, safe to keep).
        With ``reuse_buffers``: cycle the persistent ring -- pages stay warm, no
        per-window allocation -- and the returned window aliases the slot until
        the ring wraps back to it.
        """
        if not self._reuse_buffers:
            return EventSoABuffers(need)
        slots = self._dt_slots
        i = self._dt_slot_i
        if i >= len(slots):
            slots.append(EventSoABuffers(need))
        self._dt_slot_i = (i + 1) % self._dt_ring_size()
        buf = slots[i]
        if buf.capacity < need:
            buf.grow(need)
        buf.c.size = 0
        buf.c.capacity = buf.capacity
        return buf

    def _read_delta_t_c(self, delta_t: int) -> "EventArray":
        """One delta_t window via the dedicated C parser.

        The C parser stops exactly when an event's timestamp reaches ``end_ts``,
        so a whole window is decoded in one (GIL-free) call straight into the
        output buffer -- no ``searchsorted`` boundary hunt and no overshoot carry
        (every returned event already has ``ts < end_ts``). This is the openeb
        approach: the slicing lives in the decoder, not in Python.
        """
        dec = self._file_decoder
        n_cap = self._n_events
        tr = self._dt_trigger_sink()

        # Anchor the window origin on the first event once, then rewind so the C
        # parser redoes window 0 from the start with the right end_ts.
        if not self._anchored:
            first = self._peek_first_ts()
            if first is None:
                self._eof = True
                return EventArray.empty()
            self._first_ts = first
            self._current_ts = first
            self._anchored = True
            dec.reset()
            self._eof = False

        end_ts = self._current_ts + delta_t

        out = self._acquire_window_buffer(
            max(self._dt_est * 5 // 4 + self._dt_step, self._dt_step))
        # The n_events safety cap (max_events in delta_t mode) is enforced by
        # clamping the capacity the C parser sees, so a single C call can never
        # decode past it. The parser stops within its reserved vector headroom
        # below the cap; the decoder offset stays exact, so the next call
        # continues the same window with no loss.
        out.c.capacity = min(out.capacity, n_cap)
        count_cut = False
        while True:
            tr.reset()
            _, status = dec.parse_step_delta_t(out, tr, end_ts)
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
                new_cap = min(max(out.size + self._dt_step,
                                  int(out.c.capacity * 1.5) + 1), n_cap)
                out.grow(new_cap)          # realloc only if backing too small
                out.c.capacity = new_cap   # parser-visible cap (recycled slots)
            # else EVUTILS_PARSE_OK: input drained this call; loop (tail -> EOF).

        size = out.size
        out.c.size = size
        self._dt_est = max(self._dt_est, size)
        # On a normal window advance the clock; on a count-cut stay in the same
        # time window so the next call continues it (matches the accumulator path).
        if not count_cut:
            self._current_ts += delta_t
        self._n_read_events += size
        self._eof = dec.is_eof()

        output = events_view(out)
        if self._normalize_ts:
            output.t -= self._first_ts - self._start_ts
        return output

    def _flush_dt_carry(self) -> None:
        """Fold a pending delta_t fast-path overshoot into the accumulator.

        The fast paths keep up to one decode step of overshoot in
        ``_dt_carry``; any code path that reads via the accumulator (mixed-mode
        overrides, ``read_all``) must fold it in first or those events are
        silently lost. Allocates the accumulator if needed.
        """
        carry = self._dt_carry
        if carry is None or carry.size == 0:
            self._dt_carry = None
            return
        if self._buffer is None:
            self._buffer = EventAccumulator(self._acc_capacity)
        self._buffer.append(events_view(carry))
        self._dt_carry = None
        # Carried events exist => stream is not exhausted for the reader even
        # if the decoder itself hit EOF.
        self._eof = False

    def _attach_sensor_size(self, out: "EventArray") -> "EventArray":
        """Stamp ``sensor_size=(width, height)`` onto returned events, when known.

        Applied at the single internal read chokepoint so every delivery path
        (``read``, ``read_all``, sync and async iteration) carries the sensor
        geometry. Triggers are left untouched. Formats that do not expose a
        geometry (``shape()`` returns ``None``) leave ``sensor_size`` as ``None``.
        """
        w, h = self._file_decoder.shape()
        if w is None or h is None:
            return out
        ev = out[0] if isinstance(out, tuple) else out
        try:
            ev.sensor_size = (int(w), int(h))
        except AttributeError:
            pass
        return out

    def _read(self, delta_t:int|None=None, n_events:int|None=None) -> "EventArray":
        """Thin wrapper stamping sensor_size onto the decoded window."""
        return self._attach_sensor_size(self._read_impl(delta_t, n_events))

    def _read_impl(self, delta_t:int|None=None, n_events:int|None=None) -> "EventArray":
        """Unguarded body of :meth:`read` (also driven by the prefetch worker)."""
        # If not initialized, initialize
        if not self._is_initialized:
            self.init()

        # Fast path: pure n_events streaming with no time window, no triggers,
        # and nothing left buffered from an earlier windowed read. Skips the
        # staging accumulator (and its per-window copy).
        if (self._mode == "n_events" and delta_t is None
                and not self._read_external_triggers
                and (self._buffer is None or len(self._buffer) == 0)):
            dec = self._file_decoder
            n = n_events if n_events is not None else self._n_events
            # Exact-capacity native decoders (EVT2/DAT/AER) decode straight into
            # a fresh output buffer via parse_step.
            if self._native_fill and dec._exact_window:
                return self._read_n_events_fast(n)
            # Decoders whose read_chunk already returns independent, bounded
            # chunks (NPZ/CSV) can hand that out directly.
            if dec._independent_windows:
                return self._read_n_events_readchunk(n)

        # Fast path: pure delta_t streaming with a native-fill decoder, no
        # triggers, and nothing left in the staging accumulator. Hands out a
        # zero-copy view of the window instead of copying it out (slice_copy).
        if (self._mode == "delta_t" and n_events is None
                and self._native_fill
                and not self._read_external_triggers
                and (self._buffer is None or len(self._buffer) == 0)):
            dt = delta_t if delta_t is not None else self._delta_t
            # Prefer the dedicated C delta_t parser (one GIL-free call per window,
            # no boundary search or overshoot carry) when the format has one.
            if self._file_decoder._has_delta_t_parser:
                return self._read_delta_t_c(dt)
            return self._read_delta_t_fast(dt)

        # Allocate the staging accumulator on first use.
        if self._buffer is None:
            self._buffer = EventAccumulator(self._acc_capacity)
        acc = self._buffer

        # If a delta_t fast path ran earlier and left an overshoot carry, fold
        # it in first: those events precede anything the decoder emits next.
        # (Mixed-path reads -- per-call overrides after windowed fast-path
        # iteration -- would otherwise silently lose them.)
        self._flush_dt_carry()

        # "all" mode (without per-call overrides) means the whole remaining
        # stream: skip the window cutoffs entirely and drain to EOF, so files
        # exceeding max_time/max_events are still returned in full.
        drain = self._mode == "all" and delta_t is None and n_events is None

        # Override the parameters if they are specified
        if delta_t is None:
            delta_t = self._delta_t
        if n_events is None:
            n_events = self._n_events

        # Establish the first timestamp once, at the very start of the stream.
        if not self._anchored and len(acc) == 0:
            if self._pull(acc, delta_t, n_events) == 0:
                self._eof = True
                if self._read_external_triggers:
                    from ..types import TriggerArray
                    return EventArray.empty(), TriggerArray.empty()
                return EventArray.empty()
            self._first_ts = int(acc.t_window()[0])
            self._current_ts = self._first_ts
            self._anchored = True

        start_ts: int = self._current_ts
        end_ts: int = start_ts + delta_t  # Final end_ts if we reach delta_t
        end_idx: int = len(acc)

        tr_end_idx: int = acc._tr.size - acc._tr_start

        # Gather events until we hit the n_events count, the delta_t time window,
        # or the end of the stream. Work directly on the SoA `t` column.
        # Both cutoffs are evaluated together: whichever falls earlier in the
        # stream wins (the buffer may hold far more than one window's worth).
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
                    self._current_ts = int(t[n_events])
                    tr_end_idx = int(np.searchsorted(tr_t, self._current_ts, side='left'))
                else:
                    # delta_t cutoff comes first (ties go to the time window)
                    end_idx = time_idx
                    self._current_ts += delta_t
                    tr_end_idx = int(np.searchsorted(tr_t, end_ts, side='left'))
                break

            # Not enough buffered yet: pull more from the decoder.
            if self._pull(acc, delta_t, n_events) == 0:
                self._eof = True
                # Neither cutoff met, so the whole remaining buffer is the slice.
                end_idx = len(acc)
                tr_end_idx = acc._tr.size - acc._tr_start
                break

        # Copy out the window (independent) and advance past it.
        output, output_tr = acc.slice_copy(end_idx, tr_end_idx)
        self._n_read_events += end_idx

        if self._normalize_ts:
            # Normalize the timestamps to start from zero at start_ts. slice_copy
            # already returned an independent array, so this is safe in place.
            output.t -= self._first_ts - self._start_ts
            if len(output_tr) > 0:
                output_tr.t -= self._first_ts - self._start_ts

        if self._read_external_triggers:
            return output, output_tr
        return output

    def read_all(self) -> 'EventArray | tuple[EventArray, TriggerArray]':
        """Decode and return every remaining event at once.

        Delegates to the decoder's :meth:`~evutils.io.common.EventDecoder.read_all`,
        which (for the native EVT/DAT/AER decoders) decodes the whole payload
        straight into a single output buffer -- no per-chunk copy, no final
        ``concatenate`` -- and hands the columns back as a zero-copy
        :class:`EventArray`. This bypasses the slicing ring buffer that
        :meth:`read` uses for ``delta_t``/``n_events`` windowing.

        .. note::
            This materialises every event in memory at once. For recordings too
            large to fit, iterate the reader (windowed :meth:`read`) instead.

        Returns
        -------
        EventArray
            All remaining events.

        Examples
        --------
        >>> reader = EventReader(b"% format EVT3;height=720;width=1280\\n% end\\n")
        >>> all_events = reader.read_all()
        >>> print(f"Total events: {len(all_events)}")
        Total events: 0

        """
        self._check_no_active_prefetch()
        if not self._is_initialized:
            self.init()

        # Fold in any delta_t fast-path overshoot so it is prepended below
        # along with the rest of the staging buffer.
        self._flush_dt_carry()

        _out = self._file_decoder.read_all()
        if self._read_external_triggers:
            if isinstance(_out, tuple):
                out, out_tr = _out
            else:
                out = _out
                from ..types import TriggerArray
                out_tr = TriggerArray.empty()
        else:
            out = _out # type: ignore

        # Prepend anything already buffered by prior read() calls (rare: only if
        # read() and read_all() are mixed on the same reader).
        if self._buffer is not None and (len(self._buffer) > 0 or self._buffer._tr.size - self._buffer._tr_start > 0):
            buffered, buffered_tr = self._buffer.slice_copy(len(self._buffer), self._buffer._tr.size - self._buffer._tr_start)
            if len(out) == 0:
                out = buffered
            elif len(buffered) > 0:
                out = EventArray(
                    np.concatenate([buffered.t, out.t]),
                    np.concatenate([buffered.x, out.x]),
                    np.concatenate([buffered.y, out.y]),
                    np.concatenate([buffered.p, out.p]),
                )

            if self._read_external_triggers:
                if len(out_tr) == 0:
                    out_tr = buffered_tr
                elif len(buffered_tr) > 0:
                    from ..types import TriggerArray
                    out_tr = TriggerArray(
                        np.concatenate([buffered_tr.t, out_tr.t]),
                        np.concatenate([buffered_tr.p, out_tr.p]),
                        np.concatenate([buffered_tr.id, out_tr.id]),
                    )

        self._eof = True
        self._n_read_events += len(out)
        self._attach_sensor_size(out)

        if self._normalize_ts and len(out) > 0:
            # Capture the shift before modifying out.t, so triggers get the
            # same normalization.
            shift = int(out.t[0]) - self._start_ts
            out.t -= shift
            if self._read_external_triggers and len(out_tr) > 0:
                out_tr.t -= shift

        if self._batch_mode:
            from ..types import DataBatch, TriggerArray
            if self._read_external_triggers:
                return DataBatch(events=out, triggers=out_tr)
            return DataBatch(events=out, triggers=TriggerArray.empty())
        if self._read_external_triggers:
            return out, out_tr
        return out

    def reset(self) -> None:
        """Reset file reader back to the beginning of the file.

        An active asynchronous iterator is cancelled first (its remaining
        buffered chunks are discarded).
        """
        if self._active_prefetch is not None:
            self._active_prefetch.close()
        self._n_read_events = 0
        self._eof = False
        self._anchored = False
        self._pace_anchor = None
        self._dt_carry = None
        self._dt_est = self._step
        self._dt_slot_i = 0  # keep the recycled slots themselves (warm pages)
        if self._buffer is not None:
            self._buffer.reset()
        self._file_decoder.reset()

    def seek(self, t: int | None = None, n: int | None = None,
             relative: bool = False) -> int:
        """Reposition the read cursor by timestamp or event index.

        Random access in *event coordinates*: give exactly one of ``t``
        (absolute microseconds) or ``n`` (absolute 0-based event index). With
        ``relative=True`` the value is added to the current cursor instead
        (``whence=CUR``). Seeks work both forward and backward. After a seek the
        next :meth:`read` / iteration continues in the reader's configured
        ``delta_t`` / ``n_events`` mode, re-anchored at the new point.

        A seekable source uses the decoder's index / binary search (fast); a
        non-seekable one falls back to iterating and dropping events up to the
        target (backward requires restarting the stream).

        Parameters
        ----------
        t : int, optional
            Target timestamp in microseconds.
        n : int, optional
            Target event index (0-based).
        relative : bool, optional
            Interpret ``t`` / ``n`` relative to the current cursor.

        Returns
        -------
        int
            The absolute timestamp of the first event that will be read next.

        Raises
        ------
        NotImplementedError
            If the decoder does not support seeking.
        ValueError
            If neither or both of ``t`` / ``n`` are given.

        Examples
        --------
        >>> reader = EventReader(b"% format EVT3;height=720;width=1280\\n% end\\n", delta_t=10000)
        >>> _ = reader.seek(t=0)
        """
        if not self._is_initialized:
            self.init()
        if (t is None) == (n is None):
            raise ValueError("seek() requires exactly one of t= or n=.")

        dec = self._file_decoder
        if not dec.SUPPORTS_SEEK:
            raise NotImplementedError(
                f"{dec.__class__.__name__} does not support seeking."
            )

        if self._active_prefetch is not None:
            self._active_prefetch.close()

        if relative:
            if t is not None:
                t = self._current_ts + t
            else:
                n = self._n_read_events + n

        # Anchor the normalization origin (_first_ts) on the stream's *first*
        # event before the cursor moves, so normalize_ts is call-order
        # independent (seek-then-read == read-then-seek). Peeking rewinds the
        # decoder, so it is skipped on non-seekable sources (and when
        # normalization is off) -- there the origin falls back to the seek
        # landing point below.
        first_ts: int | None = None
        seekable = getattr(self._source, "seekable", lambda: True)()
        if not self._anchored and self._normalize_ts and seekable:
            first_ts = self._peek_stream_first_ts()

        self._eof = False
        self._pace_anchor = None
        self._dt_carry = None
        self._dt_est = self._step
        self._dt_slot_i = 0
        if self._buffer is not None:
            self._buffer.reset()

        if seekable:
            try:
                res, rem_ev, rem_tr = dec.seek(t=t, n=n)
            except (io.UnsupportedOperation, OSError):
                res, rem_ev, rem_tr = self._seek_linear(t, n)
        else:
            res, rem_ev, rem_tr = self._seek_linear(t, n)

        if len(rem_ev) > 0 or (rem_tr is not None and len(rem_tr) > 0):
            if self._buffer is None:
                self._buffer = EventAccumulator(self._acc_capacity)
            self._buffer.append(rem_ev, rem_tr)

        if not self._anchored:
            self._first_ts = first_ts if first_ts is not None else res.ts
        self._n_read_events = res.index if res.index >= 0 else (int(n) if n is not None else 0)
        self._current_ts = res.ts
        self._anchored = True
        return res.ts

    def _peek_stream_first_ts(self) -> int | None:
        """Timestamp of the stream's very first event, leaving the decoder
        rewound to the start.

        Used to anchor the ``normalize_ts`` origin before a seek moves the
        cursor. Only called before any read on a seekable source (rewinding a
        non-seekable decoder would lose data). Returns ``None`` for an empty
        stream.
        """
        dec = self._file_decoder
        dec.reset()
        first: int | None
        if self._native_fill:
            first = self._peek_first_ts()
        else:
            chunk = dec.read_chunk()
            if isinstance(chunk, tuple):
                chunk = chunk[0]
            first = int(chunk.t[0]) if len(chunk) > 0 else None
        dec.reset()
        return first

    def _seek_linear(self, t: int | None, n: int | None) -> tuple["SeekResult", "EventArray", "TriggerArray | None"]:
        """Fallback seek for non-seekable sources: iterate and drop to target."""
        from .common import SeekResult
        dec = self._file_decoder
        axis = "t" if t is not None else "n"
        target = t if t is not None else n

        behind = ((axis == "t" and target < self._current_ts)
                  or (axis == "n" and target < self._n_read_events))
        seen = 0
        if behind:
            dec.reset()
        else:
            seen = self._n_read_events

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
                self._eof = True
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
        return SeekResult(ts=landed_ts, index=idx, eof=self._eof), rem_ev, rem_tr

    def __enter__(self) -> "EventReader":
        return self

    def is_eof(self) -> bool:
        """Check if the end of the file is reached.

        Returns
        -------
        bool
            True if the end of the file is reached, False otherwise

        """
        if not self._is_initialized:
            self.init()
        return self._eof and (self._buffer is None or len(self._buffer) == 0)

    def close(self) -> None:
        """Close the reader and release resources (decoder buffer views, then the
        underlying byte source). Cancels any active asynchronous iterator first.
        """
        if self._active_prefetch is not None:
            self._active_prefetch.close()
        # Drop decoder views (e.g. into an mmap) before closing the source.
        self._file_decoder.close()
        self._source.close()

    def __exit__(self, exc_type: "type[BaseException] | None", exc_value: "BaseException | None", traceback: "types.TracebackType | None") -> None:
        self.close()

    def __repr__(self) -> str:
        if self._is_initialized:
            is_initialized_txt = "initialized"
        else:
            is_initialized_txt = "not initialized"
        src = self._file_name if self._file_name is not None else self._source.__class__.__name__
        return f"{self.__class__.__name__}(source={src} - {is_initialized_txt}, delta_t={self._delta_t}, n_events={self._n_events}, mode={self._mode})"

    def __len__(self) -> int:
        return self._n_read_events

    def __iter__(self) -> "Iterator[EventArray]":
        """Iterate over the events in the file.

        With ``async_read=True`` the windows are decoded ahead in a background
        thread (bounded to a couple of windows), overlapping decode with the
        caller's per-chunk processing. Only one asynchronous iterator can be
        active per reader; direct :meth:`read` / :meth:`read_all` calls raise
        until it is exhausted or closed.

        Yields
        ------
        EventArray
            An array with the events

        Examples
        --------
        >>> reader = EventReader(b"% format EVT3;height=720;width=1280\\n% end\\n", n_events=5000)
        >>> for events in reader:
        ...     print(f"Received chunk of {len(events)} events")
        Received chunk of 0 events

        """
        it: Any
        if self._async_read:
            self._check_no_active_prefetch()
            kwargs: dict[str, str | int | float | bool] = {}
            if self._prefetch_depth is not None:
                kwargs["depth"] = self._prefetch_depth
            it = PrefetchIterator(
                self._iter_sync(),
                on_finish=lambda: setattr(self, "_active_prefetch", None),
                **kwargs,
            )
            self._active_prefetch = it
        else:
            it = self._iter_sync()
        if self._real_time:
            # Pace on the consumer side, never inside the decode path: with
            # async_read the worker keeps decoding ahead while we sleep.
            return self._paced_iter(it)
        return it

    def _paced_iter(self, it: "Iterator[EventArray]") -> "Iterator[EventArray]":
        """Wrap an iterator so each chunk is released on the playback clock."""
        try:
            for chunk in it:
                self._pace(chunk)
                yield chunk
        finally:
            # Propagate early termination (break / close) to a prefetching
            # iterator so its worker thread is released.
            if hasattr(it, "close"):
                it.close()

    def _iter_sync(self) -> "Iterator[Any]":
        """The plain synchronous window generator behind :meth:`__iter__`."""
        if not self._is_initialized:
            self.init()
        while not self.is_eof():
            res = self._read()
            if self._batch_mode:
                from ..types import DataBatch, TriggerArray
                if isinstance(res, tuple):
                    yield DataBatch(events=res[0], triggers=res[1])
                else:
                    yield DataBatch(events=res, triggers=TriggerArray.empty())
            else:
                yield res
    def shape(self) -> tuple[int|None, int|None]:
        """Get the shape of the frame.

        Returns
        -------
        tuple[int, int]
            The shape of the frame (width, height)

        """
        if not self._is_initialized:
            self.init()
        if self._width is not None and self._height is not None:
            return self._width, self._height
        else:
            return self._file_decoder.shape()

    def tell(self) -> int:
        """Get the current position in the file.

        Returns
        -------
        int
            The current position in the file

        """
        if not self._is_initialized:
            self.init()
        return self._file_decoder.tell()

