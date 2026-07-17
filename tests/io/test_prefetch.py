"""Tests for asynchronous (prefetching) iteration: EventReader(async_read=True).

The iterator must be byte-identical to synchronous iteration, propagate
worker exceptions, survive early exits without deadlocking, and guard the
reader against concurrent direct reads.
"""
import numpy as np
import pytest

from evutils.io import EventReader, EventWriter
from evutils.io._prefetch import PrefetchIterator
from evutils.types import Event_dtype


from typing import Any
@pytest.fixture()
def raw_file(tmp_path: Any) -> Any:
    rng = np.random.default_rng(7)
    n = 200_000
    ev = np.zeros(n, dtype=Event_dtype)
    ev["t"] = np.sort(rng.integers(0, 1_000_000, n))
    ev["x"] = rng.integers(0, 1280, n)
    ev["y"] = rng.integers(0, 720, n)
    ev["p"] = rng.integers(0, 2, n)
    p = tmp_path / "events.raw"
    with EventWriter(p) as w:
        w.write(ev)
    return p, ev


@pytest.mark.parametrize("suffix", [".raw", ".npz"])
def test_async_matches_sync(tmp_path: Any, raw_file: Any, suffix: str) -> None:
    """Async iteration yields the same windows as sync, for a native (C)
    decoder and a pure-Python one."""
    p, ev = raw_file
    if suffix == ".npz":
        p2 = tmp_path / "events.npz"
        with EventWriter(p2) as w:
            w.write(ev)
        p = p2

    with EventReader(p, n_events=30_000) as r:
        sync_chunks = [np.asarray(c).copy() for c in r]
    with EventReader(p, n_events=30_000, async_read=True) as r:
        async_chunks = [np.asarray(c) for c in r]

    assert len(sync_chunks) == len(async_chunks)
    for s, a in zip(sync_chunks, async_chunks):
        assert np.array_equal(s, a)


def test_async_ext_triggers(raw_file: Any) -> None:
    """(events, triggers) tuples pass through the prefetch queue unchanged."""
    p, _ = raw_file
    with EventReader(p, n_events=50_000, ext_trigger=True, async_read=True) as r:
        for ev, tr in r:
            assert len(ev) > 0
            assert hasattr(tr, "id")


def test_direct_read_guarded_while_iterating(raw_file: Any) -> None:
    p, _ = raw_file
    with EventReader(p, n_events=50_000, async_read=True) as r:
        it = iter(r)
        next(it)
        with pytest.raises(RuntimeError, match="asynchronous iterator is active"):
            r.read()
        with pytest.raises(RuntimeError, match="asynchronous iterator is active"):
            r.read_all()
        # After exhausting the iterator, direct reads are allowed again.
        for _ in it:
            pass
        assert len(r.read_all()) == 0  # EOF, but no guard error


def test_early_break_and_reset(raw_file: Any) -> None:
    """Breaking out of async iteration must not deadlock; reset() cancels the
    worker and a fresh (async) iteration sees the whole file again."""
    p, ev = raw_file
    with EventReader(p, n_events=10_000, async_read=True) as r:
        for i, _ in enumerate(r):
            if i == 1:
                break
        r.reset()  # cancels the active iterator
        total = sum(len(c) for c in r)
    assert total == len(ev)


def test_close_with_active_iterator(raw_file: Any) -> None:
    p, _ = raw_file
    r = EventReader(p, n_events=10_000, async_read=True)
    it = iter(r)
    next(it)
    r.close()  # must join the worker and not raise
    assert r._active_prefetch is None


def test_only_one_active_iterator(raw_file: Any) -> None:
    p, _ = raw_file
    with EventReader(p, n_events=50_000, async_read=True) as r:
        it = iter(r)
        next(it)
        with pytest.raises(RuntimeError, match="asynchronous iterator is active"):
            iter(r)
        it.close()
        r.seek(n=0)
        assert sum(len(c) for c in r) > 0  # a new iterator is fine now


def test_worker_exception_propagates() -> None:
    """An exception raised by the source surfaces in the consumer thread."""
    def broken() -> Any:
        yield 1
        yield 2
        raise ValueError("decoder blew up")

    it = PrefetchIterator(broken())
    assert next(it) == 1
    assert next(it) == 2
    with pytest.raises(ValueError, match="decoder blew up"):
        next(it)


def test_prefetch_iterator_bounded() -> None:
    """The queue never buffers more than `depth` chunks ahead."""
    import time

    produced = []

    def source() -> Any:
        for i in range(100):
            produced.append(i)
            yield i

    it = PrefetchIterator(source(), depth=2)
    time.sleep(0.3)  # give the worker every chance to run ahead
    # depth chunks buffered + one in the worker's hand at most
    assert len(produced) <= 2 + 1
    assert list(it) == list(range(100))


def test_prefetch_iterator_close_idempotent() -> None:
    it = PrefetchIterator(iter(range(10)), depth=1)
    next(it)
    it.close()
    it.close()
    with pytest.raises(StopIteration):
        next(it)
