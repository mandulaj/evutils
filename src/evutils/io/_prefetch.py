"""Background-thread prefetching for :class:`~evutils.io.EventReader`.

The reader's windowing loop runs in a worker thread and pushes finished
windows into a small bounded queue, so decoding the next window overlaps the
caller's processing of the current one. This works because

* the native parsers are called through ctypes, which releases the GIL for
  the duration of the C call (measured: a Python thread keeps ~90% of its
  solo throughput while a decode runs), and
* every yielded window is already an independent copy of the reader's reused
  accumulator storage, so the worker can never mutate chunks the consumer
  still holds.

When it helps: any pipeline where per-chunk processing takes meaningful time
-- numpy transforms, writing elsewhere, GPU inference (where the read becomes
entirely free). When it does not: processing that already saturates memory
bandwidth can get *slower* with prefetching enabled, because decode competes
for the same bandwidth. Hence prefetching is opt-in
(``EventReader(..., async_read=True)``).
"""
from __future__ import annotations

import queue
import threading
from typing import Any, Callable, Iterator

#: How many finished windows may be buffered ahead of the consumer. Depth 2 is
#: enough to decouple producer and consumer; larger values only cost memory
#: (depth x window size).
DEFAULT_DEPTH = 2


class PrefetchIterator:
    """Iterate a chunk source through a bounded queue filled by a worker thread.

    Semantics match plain iteration: same chunks in the same order, and an
    exception raised by the source is re-raised to the consumer at the point
    it would have occurred. Use :meth:`close` (or exhaust the iterator, or use
    it as a context manager) to release the worker early; an abandoned,
    unclosed iterator does not deadlock -- the worker is a daemon thread
    parked on a stop-aware put.

    Parameters
    ----------
    source : Iterator
        The synchronous chunk iterator to drain (runs entirely in the worker
        thread; it must not be touched by anyone else while this is alive).
    depth : int
        Maximum number of chunks buffered ahead of the consumer.
    on_finish : callable, optional
        Called exactly once, from whichever thread finishes the iterator
        (exhaustion or :meth:`close`). The EventReader uses it to clear its
        active-prefetch guard.

    """

    _SENTINEL: Any = object()

    def __init__(self, source: Iterator[Any], depth: int = DEFAULT_DEPTH,
                 on_finish: Callable[[], None] | None = None):
        if depth < 1:
            raise ValueError("prefetch depth must be >= 1")
        self._source = source
        self._queue: queue.Queue[Any] = queue.Queue(maxsize=depth)
        self._stop = threading.Event()
        self._exc: BaseException | None = None
        self._finished = False
        self._on_finish = on_finish
        self._worker = threading.Thread(
            target=self._work, name="evutils-prefetch", daemon=True
        )
        self._worker.start()

    # ------------------------------------------------------------------ #
    # Worker side
    # ------------------------------------------------------------------ #
    def _put(self, item: Any) -> bool:
        """Blocking put that stays responsive to :meth:`close`.

        Returns False when cancelled. The end-of-stream sentinel must go
        through here too: a non-blocking put of the sentinel can be dropped
        when the queue is full, leaving the consumer waiting forever.
        """
        while not self._stop.is_set():
            try:
                self._queue.put(item, timeout=0.1)
                return True
            except queue.Full:
                pass
        return False

    def _work(self) -> None:
        try:
            for chunk in self._source:
                if not self._put(chunk):
                    return  # cancelled by close()
        except BaseException as exc:  # propagated to the consumer
            self._exc = exc
        finally:
            close = getattr(self._source, "close", None)
            if callable(close):
                close()  # generators: release reader frame in this thread
            self._put(self._SENTINEL)

    # ------------------------------------------------------------------ #
    # Consumer side
    # ------------------------------------------------------------------ #
    def __iter__(self) -> "PrefetchIterator":
        return self

    def __next__(self) -> Any:
        if self._finished:
            raise StopIteration
        item = self._queue.get()
        if item is self._SENTINEL:
            self._worker.join()
            self._finish()
            if self._exc is not None:
                raise self._exc
            raise StopIteration
        return item

    def close(self) -> None:
        """Cancel the worker and drop any buffered chunks. Idempotent."""
        if self._finished:
            return
        self._stop.set()
        # Drain so a worker blocked on put() can observe the stop event.
        while True:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break
        self._worker.join(timeout=5.0)
        self._finish()

    def _finish(self) -> None:
        if not self._finished:
            self._finished = True
            if self._on_finish is not None:
                self._on_finish()

    def __enter__(self) -> "PrefetchIterator":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()
