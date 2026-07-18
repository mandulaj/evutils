"""Decomposed :class:`EventReader` facade (V2).

A thin facade over :class:`~evutils.io.v2.context.ReadContext`, the per-mode
:mod:`~evutils.io.v2.strategies`, the :class:`~evutils.io.v2.cursor.SeekCursor`,
and the :class:`~evutils.io.v2.pacing.Pacer`. It parses / validates config and
infers the read mode (ported from V1's ``__init__``), then delegates windowing,
seeking, and pacing to those components. The public surface matches V1 exactly.
"""

from __future__ import annotations

import io
import types as _types
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterator

from ...types import EventArray, TriggerArray
from .. import decoders as ev_decoders
from .._prefetch import PrefetchIterator
from .._source import ByteSource, make_source
from ..common import SeekResult
from .context import ReadContext
from .cursor import SeekCursor
from .pacing import Pacer
from .strategies import AllStrategy, select_strategy

if TYPE_CHECKING:
    pass


class EventReader:
    """Reads and automatically decodes / slices events from a file or stream.

    Decomposed reimplementation of :class:`evutils.io.EventReader` (V1): same
    public surface and semantics, but the read modes, seeking, pacing, and
    buffer recycling live in separate collaborating components instead of one
    monolith. See the V1 docstring for the full parameter reference.
    """

    READING_MODES = ["delta_t", "n_events", "mixed", "all", "auto"]
    DEFAULT_N_EVENTS = 1_000_000
    DEFAULT_DELTA_T = 10_000

    def __init__(self, file: "Path | str | io.BufferedReader | bytes | ByteSource",
                 delta_t: "int | None" = None,
                 n_events: "int | None" = None,
                 mode: str = "auto",
                 start_ts: int = 0,
                 normalize_ts: bool = False,
                 max_time: int = 1_000_000_000_000,
                 max_events: int = 10_000_000,
                 width: "int | None" = None, height: "int | None" = None,
                 ext_trigger: bool = False,
                 async_read: bool = False,
                 prefetch_depth: "int | None" = None,
                 reuse_buffers: bool = False,
                 real_time: bool = False,
                 playback_speed: float = 1.0,
                 max_gap: "float | None" = 1.0,
                 index: "str | bool" = "auto",
                 strict: bool = False,
                 batch_mode: bool = False,
                 file_decoder: "ev_decoders.EventDecoder | type[ev_decoders.EventDecoder] | None" = None,
                 **kwargs) -> None:

        self._file_name: "Path | None" = Path(file) if isinstance(file, (str, Path)) else None
        self._read_external_triggers = ext_trigger
        self._batch_mode = batch_mode

        # 1. Normalise the input into a ByteSource.
        self._source: ByteSource = make_source(file)

        # 2. Resolve and launch the decoder.
        if isinstance(file_decoder, ev_decoders.EventDecoder):
            self._file_decoder = file_decoder
        else:
            decoder_cls = file_decoder or ev_decoders.resolve_decoder_cls(self._source)
            self._file_decoder = decoder_cls(self._source, **kwargs)

        self._file_decoder.read_external_triggers = self._read_external_triggers
        self._file_decoder._strict = strict
        if self._read_external_triggers and not self._file_decoder.SUPPORTS_EXT_TRIGGERS:
            import warnings
            warnings.warn(f"{self._file_decoder.__class__.__name__} does not support reading external triggers.")

        self._eof = False
        self._width = width
        self._height = height
        self._start_ts = start_ts
        self._first_ts = 0
        self._current_ts = self._first_ts
        self._normalize_ts = normalize_ts

        # Validate the parameters + infer mode (ported verbatim from V1).
        if mode not in EventReader.READING_MODES:
            raise ValueError(f"Mode {mode} not supported. Supported modes are: {EventReader.READING_MODES}")

        self._mode = mode.lower()

        if self._mode == "auto":
            if delta_t is not None and n_events is not None:
                self._mode = "mixed"
            elif delta_t is not None:
                self._mode = "delta_t"
                n_events = max_events
            elif n_events is not None:
                self._mode = "n_events"
                delta_t = max_time
            else:
                self._mode = "mixed"
                delta_t = self.DEFAULT_DELTA_T
                n_events = self.DEFAULT_N_EVENTS
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

        self._delta_t = delta_t
        self._n_events = n_events
        self._max_events = max_events if max_events < self._n_events else self._n_events
        self._max_time = max_time if max_time < self._delta_t else self._delta_t

        self._is_initialized = False

        step = 1 << 20
        acc_capacity = min(self._n_events, 4 * step) + 2 * step
        native_fill = hasattr(self._file_decoder, "parse_step")

        # Async iteration config.
        self._async_read = async_read
        if prefetch_depth is not None and prefetch_depth < 1:
            raise ValueError("prefetch_depth must be >= 1")
        self._prefetch_depth = prefetch_depth
        self._active_prefetch: "PrefetchIterator | None" = None

        # Real-time playback config.
        if not isinstance(playback_speed, (int, float)) or playback_speed <= 0:
            raise ValueError("playback_speed must be a positive number")
        self._real_time = real_time
        self._playback_speed = float(playback_speed)
        if max_gap is not None and (not isinstance(max_gap, (int, float)) or max_gap <= 0):
            raise ValueError("max_gap must be a positive number or None")
        self._max_gap = float(max_gap) if max_gap is not None else None
        self._pacer = Pacer(self._playback_speed, self._max_gap)

        # Seek-index wiring (only EVT consults these).
        self._index_opt = index
        self._file_decoder._use_sidecar = (index == "metavision")
        self._file_decoder._persist_index = (index == "persist")
        if self._file_name is not None:
            self._file_decoder._raw_path = str(self._file_name)

        # Shared cross-cutting read state.
        self.ctx = ReadContext(
            decoder=self._file_decoder,
            normalize_ts=self._normalize_ts,
            start_ts=self._start_ts,
            first_ts=self._first_ts,
            current_ts=self._current_ts,
            delta_t=self._delta_t,
            n_events=self._n_events,
            read_external_triggers=self._read_external_triggers,
            acc_capacity=acc_capacity,
            native_fill=native_fill,
            step=step,
            dt_est=step,
            reuse_buffers=bool(reuse_buffers),
            async_read=async_read,
            prefetch_depth=prefetch_depth,
        )

        self._cursor = SeekCursor(self, self.ctx)

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #
    def init(self) -> None:
        """Initialize the reader (implicitly done on first read)."""
        if self._is_initialized:
            return
        self._file_decoder.init()
        self._is_initialized = True

    def _check_no_active_prefetch(self) -> None:
        if self._active_prefetch is not None:
            raise RuntimeError(
                "An asynchronous iterator is active on this reader: exhaust or "
                "close() it before calling read()/read_all(), or iterate instead."
            )

    # ------------------------------------------------------------------ #
    # Reading
    # ------------------------------------------------------------------ #
    def _attach_sensor_size(self, out: "EventArray") -> "EventArray":
        """Stamp ``sensor_size=(width, height)`` onto returned events, when known."""
        w, h = self._file_decoder.shape()
        if w is None or h is None:
            return out
        ev = out[0] if isinstance(out, tuple) else out
        try:
            ev.sensor_size = (int(w), int(h))
        except AttributeError:
            pass
        return out

    def _read_impl(self, delta_t: "int | None" = None, n_events: "int | None" = None
                   ) -> "EventArray | tuple[EventArray, TriggerArray]":
        """Unguarded body of :meth:`read` (also driven by the prefetch worker)."""
        if not self._is_initialized:
            self.init()
        strat = select_strategy(self.ctx, self._mode, delta_t, n_events)
        return strat.next_window(delta_t, n_events)

    def _read(self, delta_t: "int | None" = None, n_events: "int | None" = None
              ) -> "EventArray | tuple[EventArray, TriggerArray]":
        """Thin wrapper stamping sensor_size onto the decoded window."""
        return self._attach_sensor_size(self._read_impl(delta_t, n_events))

    def read(self, delta_t: "int | None" = None, n_events: "int | None" = None
             ) -> "EventArray | tuple[EventArray, TriggerArray]":
        """Read one window based on the mode / parameters."""
        self._check_no_active_prefetch()
        out = self._read(delta_t, n_events)
        if self._real_time:
            self._pacer.pace(out)
        if self._batch_mode:
            from ...types import DataBatch
            if isinstance(out, tuple):
                return DataBatch(events=out[0], triggers=out[1])
            return DataBatch(events=out, triggers=TriggerArray.empty())
        return out

    def read_all(self) -> "EventArray | tuple[EventArray, TriggerArray]":
        """Decode and return every remaining event at once."""
        self._check_no_active_prefetch()
        if not self._is_initialized:
            self.init()
        out = AllStrategy(self.ctx).read_all()
        self._attach_sensor_size(out)
        if self._batch_mode:
            from ...types import DataBatch
            if self._read_external_triggers:
                return DataBatch(events=out[0], triggers=out[1])
            return DataBatch(events=out, triggers=TriggerArray.empty())
        return out

    # ------------------------------------------------------------------ #
    # Iteration
    # ------------------------------------------------------------------ #
    def __iter__(self) -> "Iterator[EventArray]":
        """Iterate over the windows in the file (optionally async / paced)."""
        it: Any
        if self._async_read:
            self._check_no_active_prefetch()
            kwargs: "dict[str, str | int | float | bool]" = {}
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
            return self._pacer.paced_iter(it)
        return it

    def _iter_sync(self) -> "Iterator[Any]":
        """The plain synchronous window generator behind :meth:`__iter__`."""
        if not self._is_initialized:
            self.init()
        while not self.is_eof():
            res = self._read()
            if self._batch_mode:
                from ...types import DataBatch
                if isinstance(res, tuple):
                    yield DataBatch(events=res[0], triggers=res[1])
                else:
                    yield DataBatch(events=res, triggers=TriggerArray.empty())
            else:
                yield res

    # ------------------------------------------------------------------ #
    # Seeking
    # ------------------------------------------------------------------ #
    def seek(self, t: "int | None" = None, n: "int | None" = None,
             relative: bool = False) -> SeekResult:
        """Reposition the read cursor by timestamp or event index."""
        return self._cursor.seek(t=t, n=n, relative=relative)

    @property
    def last_seek(self) -> "SeekResult | None":
        """The :class:`SeekResult` of the most recent :meth:`seek`, or ``None``."""
        return self._cursor.last_seek

    @property
    def event_index(self) -> int:
        """Current 0-based event index (events read so far / seek landing)."""
        return self.ctx.n_read_events

    # ------------------------------------------------------------------ #
    # State / lifecycle
    # ------------------------------------------------------------------ #
    def reset(self) -> None:
        """Reset the reader back to the beginning of the file."""
        if self._active_prefetch is not None:
            self._active_prefetch.close()
        ctx = self.ctx
        ctx.n_read_events = 0
        ctx.eof = False
        ctx.anchored = False
        ctx.dt_carry = None
        ctx.dt_est = ctx.step
        ctx.dt_slot_i = 0  # keep the recycled slots themselves (warm pages)
        self._pacer.reset()
        if ctx.accumulator is not None:
            ctx.accumulator.reset()
        self._file_decoder.reset()

    def is_eof(self) -> bool:
        """True once the stream is drained and nothing remains buffered."""
        if not self._is_initialized:
            self.init()
        acc = self.ctx.accumulator
        return self.ctx.eof and (acc is None or len(acc) == 0)

    def close(self) -> None:
        """Close the reader and release resources."""
        if self._active_prefetch is not None:
            self._active_prefetch.close()
        self._file_decoder.close()
        self._source.close()

    def shape(self) -> "tuple[int | None, int | None]":
        """Get the (width, height) of the frame."""
        if not self._is_initialized:
            self.init()
        if self._width is not None and self._height is not None:
            return self._width, self._height
        return self._file_decoder.shape()

    def tell(self) -> int:
        """Current byte position in the file."""
        if not self._is_initialized:
            self.init()
        return self._file_decoder.tell()

    def __enter__(self) -> "EventReader":
        return self

    def __exit__(self, exc_type: "type[BaseException] | None",
                 exc_value: "BaseException | None",
                 traceback: "_types.TracebackType | None") -> None:
        self.close()

    def __len__(self) -> int:
        return self.ctx.n_read_events

    def __repr__(self) -> str:
        is_initialized_txt = "initialized" if self._is_initialized else "not initialized"
        src = self._file_name if self._file_name is not None else self._source.__class__.__name__
        return (f"{self.__class__.__name__}(source={src} - {is_initialized_txt}, "
                f"delta_t={self._delta_t}, n_events={self._n_events}, mode={self._mode})")
