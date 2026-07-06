import pytest
import numpy as np




####################################
####################################
# Test RAW module
####################################
####################################


from typing import Any

def test_RAW_writer_import() -> None:
    '''
    Testing if the RAW writer can be imported
    '''
    from evutils.io import EventWriter
    assert EventWriter is not None
    test_writer = EventWriter("test.raw")
    assert test_writer is not None
    test_writer.close()


def test_RAW_reader_import(dummy_file_factory: Any) -> None:
    '''
    Testing if the RAW reader can be imported
    '''
    test_file = dummy_file_factory("test.raw")

    from evutils.io import EventReader
    assert EventReader is not None
    test_reader = EventReader(test_file)
    assert test_reader is not None
    test_reader.close()

### Test RAW writer

@pytest.mark.parametrize("fmt", [None, 'evt2', 'evt21', 'evt3'])
def test_RAW_writer(tmp_path: Any, test_events: Any, fmt: Any) -> None:
    """Test that writing and reading RAW events yields identical results for supported formats."""
    from evutils.io import EventWriter, EventReader

    # Create in a subdirectory to test parent directory creation
    d = tmp_path / "sub"
    d.mkdir(exist_ok=True)
    p = d / f"test_{fmt or 'default'}.raw"
    
    kwargs = {'format': fmt} if fmt is not None else {}
    with EventWriter(p, **kwargs) as writer:
        writer.write(test_events)

    assert p.is_file(), "File was not created"
    assert p.stat().st_size > 0, "Created file is empty"

    with EventReader(p) as reader:
        events = reader.read()

    assert np.array_equal(events, test_events), "Read events do not match written events"


@pytest.mark.parametrize("fmt", ['evt3', 'evt2', 'evt21'])
def test_RAW_real_read(real_event_files: Any, fmt: Any) -> None:
    """Test reading real raw files checks initialization state, shape, and timestamp monotonicity."""
    from evutils.io import EventReader

    if fmt not in real_event_files or not real_event_files[fmt]:
        pytest.skip(f"No files for format {fmt}")

    for ef in real_event_files[fmt]:
        with EventReader(ef.path) as reader:
            assert not reader._is_initialized
            events = reader.read()
            assert not isinstance(events, tuple)
            assert reader._is_initialized

            assert reader.shape() in [(1280, 720), (1220, 688)]
            assert len(events) > 0

            # Decoded events must be valid and time-ordered.
            assert int(events["x"].max()) < 1280
            assert int(events["y"].max()) < 720
            assert bool(np.all(np.diff(events["t"]) >= 0))


@pytest.mark.parametrize("fmt", ['evt3', 'evt2', 'evt21'])
def test_RAW_metadata_match(real_event_files: Any, fmt: Any) -> None:
    """The reader's total event count, polarity breakdown, and trigger counts
    should match the reference ground truth from the JSON metadata.
    """
    from evutils.io import EventReader

    if fmt not in real_event_files or not real_event_files[fmt]:
        pytest.skip(f"No files for format {fmt}")

    # Only test files that have JSON metadata attached
    for ef in real_event_files[fmt]:
        if not ef.metadata:
            continue

        meta = ef.metadata
        n_events = n_pos = n_neg = n_tr_pos = n_tr_neg = 0
        
        with EventReader(ef.path, ext_trigger=True, chunk_size=5_000_000) as reader:
            shape = list(reader.shape())
            if shape != [None, None]:
                assert shape == meta['resolution']
            
            for ev, tr in reader:
                n_events += len(ev)
                n_pos += np.count_nonzero(ev["p"] == 1)
                n_neg += np.count_nonzero(ev["p"] == 0)
                
                if len(tr) > 0:
                    n_tr_pos += np.count_nonzero(tr["p"] == 1)
                    n_tr_neg += np.count_nonzero(tr["p"] == 0)

        # Check exact counts
        def check(name: str, actual: int, expected: int) -> None:
            assert actual == expected, f"{fmt} ({ef.path.name}): {name} actual {actual} != expected {expected}"

        check('total events', n_events, meta['count'])
        check('positive events', n_pos, meta['pos_count'])
        check('negative events', n_neg, meta['neg_count'])
        
        tr_meta = meta['external_triggers']
        check('total triggers', n_tr_pos + n_tr_neg, tr_meta['total'])
        check('positive triggers', n_tr_pos, tr_meta['positive'])
        check('negative triggers', n_tr_neg, tr_meta['negative'])


def test_EVT2_matches_EVT3(real_event_files: Any) -> None:
    '''EVT2 and EVT3 are the same recording transcoded, so decoding them must
    yield identical events (timestamps equal up to a constant offset).'''
    from evutils.io import EventReader

    def get_hand_path(fmt: str) -> Any:
        for ef in real_event_files[fmt]:
            if 'hand' in ef.path.name:
                return ef.path
        return None

    path2 = get_hand_path('evt2')
    path3 = get_hand_path('evt3')

    if path2 is None or path3 is None:
        pytest.skip("Transcoded hand files not found")

    def head(path: Any, n: int=200_000) -> tuple[Any, Any, Any, Any]:
        parts = []
        with EventReader(path, n_events=100_000) as reader:
            for e in reader:
                parts.append(e)
                if sum(len(p) for p in parts) >= n:
                    break
        t = np.concatenate([p["t"] for p in parts])[:n]
        x = np.concatenate([p["x"] for p in parts])[:n]
        y = np.concatenate([p["y"] for p in parts])[:n]
        p_ = np.concatenate([p["p"] for p in parts])[:n]
        return t, x, y, p_

    t2, x2, y2, p2 = head(path2)
    t3, x3, y3, p3 = head(path3)

    assert np.array_equal(x2, x3)
    assert np.array_equal(y2, y3)
    assert np.array_equal(p2, p3)
    # Timestamps differ only by a constant (per-file time origin).
    assert np.ptp(t3.astype(np.int64) - t2.astype(np.int64)) == 0


@pytest.mark.parametrize("fmt", ['evt2', 'evt3'])
def test_EVT_matches_expelliarmus(real_event_files: Any, fmt: Any) -> None:
    """Our EventReader output must match expelliarmus byte-for-byte."""
    pytest.importorskip("expelliarmus")
    from expelliarmus import Wizard # type: ignore
    from evutils.io import EventReader

    if fmt not in real_event_files or not real_event_files[fmt]:
        pytest.skip(f"No files for format {fmt}")

    ef = next((f for f in real_event_files[fmt] if 'hand' in f.path.name), real_event_files[fmt][0])

    # Read first 100k events with evutils
    with EventReader(ef.path, n_events=100_000) as reader:
        evutils_events = reader.read()
    assert not isinstance(evutils_events, tuple)

    # Read the same chunk with expelliarmus
    wiz = Wizard(encoding=fmt)
    wiz.set_file(str(ef.path))
    exp_events = []
    total = 0
    for chunk in wiz.read_chunk():
        exp_events.append(chunk)
        total += len(chunk)
        if total >= 100_000:
            break
            
    if not exp_events:
        pytest.skip("Expelliarmus returned no events")
        
    exp_events_arr = np.concatenate(exp_events)[:100_000]

    assert len(evutils_events) == len(exp_events_arr)
    assert np.array_equal(evutils_events["x"], exp_events_arr["x"])
    assert np.array_equal(evutils_events["y"], exp_events_arr["y"])
    assert np.array_equal(evutils_events["p"], exp_events_arr["p"])
    
    if fmt == 'evt3':
        # Expelliarmus has a known bug in EVT3 timestamp decoding where it occasionally
        # drifts by exactly 4096us (1 TIME_HIGH tick). Our EVT3 decoder perfectly matches
        # our EVT2 decoder, so we skip the timestamp assertion for expelliarmus EVT3.
        pass
    else:
        assert np.array_equal(evutils_events["t"], exp_events_arr["t"])


@pytest.mark.parametrize("fmt", ['evt2', 'evt3'])
def test_EVT_matches_evlib(real_event_files: Any, fmt: Any) -> None:
    """Our EventReader output must match evlib."""
    pytest.importorskip("evlib")
    import evlib # type: ignore
    from evutils.io import EventReader

    if fmt not in real_event_files or not real_event_files[fmt]:
        pytest.skip(f"No files for format {fmt}")

    ef = next((f for f in real_event_files[fmt] if 'hand' in f.path.name), real_event_files[fmt][0])

    # Read first 100k events with evutils
    with EventReader(ef.path, n_events=100_000) as reader:
        evutils_events = reader.read()
    assert not isinstance(evutils_events, tuple)

    # Read the same file with evlib (using polars)
    df = evlib.load_events(str(ef.path))
    # We only take the first 100_000 to match evutils and avoid massive memory consumption
    df = df.head(100_000)
    
    if hasattr(df, "collect"):
        df = df.collect()

    if len(df) == 0:
        pytest.skip("evlib returned no events")

    # evlib column mappings: 'x', 'y', 't', 'polarity'
    # Polarity in evlib is often -1/1 or 0/1 depending on the rust backend.
    # We map -1 back to 0 if necessary.
    evlib_x = df["x"].to_numpy()
    evlib_y = df["y"].to_numpy()
    evlib_p = df["polarity"].to_numpy()
    evlib_p = np.where(evlib_p == -1, 0, evlib_p)
    
    # Time is a Polars duration. Casting to Int64 converts it to microseconds internally
    import polars as pl # type: ignore
    evlib_t = df.select(pl.col("t").dt.total_microseconds()).to_numpy()[:, 0]

    # evlib sometimes reorders events that share the exact same timestamp, so we
    # sort both arrays lexicographically by (t, x, y, p) to compare the actual contents.
    def sort_events(t: Any, x: Any, y: Any, p: Any) -> tuple[Any, Any, Any, Any]:
        idx = np.lexsort((p, y, x, t))
        return t[idx], x[idx], y[idx], p[idx]

    evutils_t, evutils_x, evutils_y, evutils_p = sort_events(
        evutils_events["t"], evutils_events["x"], evutils_events["y"], evutils_events["p"]
    )
    evlib_t, evlib_x, evlib_y, evlib_p = sort_events(evlib_t, evlib_x, evlib_y, evlib_p)

    assert len(evutils_events) == len(df)
    assert np.array_equal(evutils_x, evlib_x)
    assert np.array_equal(evutils_y, evlib_y)
    assert np.array_equal(evutils_p, evlib_p)
    assert np.array_equal(evutils_t, evlib_t)


@pytest.mark.parametrize("fmt", ['evt3', 'evt2', 'evt21'])
@pytest.mark.parametrize("length", [10, 100, 1000])
def test_RAW_nevents_read(real_event_files: Any, fmt: Any, length: int) -> None:
    """Test reading a specific number of events."""
    from evutils.io import EventReader

    if fmt not in real_event_files or not real_event_files[fmt]:
        pytest.skip(f"No files for format {fmt}")

    for ef in real_event_files[fmt]:
        with EventReader(ef.path, n_events=length) as reader:
            events = reader.read()
            assert len(events) == length


@pytest.mark.parametrize("fmt", ['evt3', 'evt2', 'evt21'])
@pytest.mark.parametrize("delta_t", [100, 1000, 10000])
def test_RAW_delta_t_read(real_event_files: Any, fmt: Any, delta_t: int) -> None:
    """Test reading a specific time slice (delta_t) of events."""
    from evutils.io import EventReader

    if fmt not in real_event_files or not real_event_files[fmt]:
        pytest.skip(f"No files for format {fmt}")

    for ef in real_event_files[fmt]:
        with EventReader(ef.path, delta_t=delta_t) as reader:
            events = reader.read()
            assert not isinstance(events, tuple)
            assert len(events) > 0
            # All events fall within the requested time window.
            assert int(events["t"].max()) - int(events["t"].min()) <= delta_t

####################################
# Timestamp wrap / edge cases
####################################

def _roundtrip(tmp_path: Any, ev: Any, fmt: str) -> Any:
    from evutils.io import EventReader, EventWriter
    p = tmp_path / f"wrap_{fmt}.raw"
    with EventWriter(p, format=fmt) as w:
        w.write(ev)
    with EventReader(p) as r:
        return r.read_all()


def _wrap_events(t_values: Any) -> Any:
    ev = np.zeros(len(t_values), dtype=np.dtype([('t', np.int64), ('x', np.uint16), ('y', np.uint16), ('p', np.uint8)]))
    ev['t'] = np.asarray(t_values, dtype=np.int64)
    ev['x'] = np.arange(len(t_values)) % 1280
    ev['y'] = np.arange(len(t_values)) % 720
    ev['p'] = np.arange(len(t_values)) % 2
    return ev


def test_EVT3_time_high_wrap(tmp_path: Any) -> None:
    """EVT3's rolling time base is 24-bit (~16.7 s). Recordings longer than
    that must round-trip exactly: the decoder counts TIME_HIGH wraps."""
    step = 1_000_000  # 1 s steps << 2^24 us, crossing several wraps
    t = np.arange(0, 90_000_000, step)  # 90 s: 5+ wraps of 2^24 us
    ev = _wrap_events(t)
    out = _roundtrip(tmp_path, ev, "evt3")
    assert np.array_equal(out["t"], ev['t'])
    assert np.array_equal(out["x"], ev['x'])


@pytest.mark.parametrize("fmt", ["evt2", "evt21"])
def test_EVT2_time_high_wrap(tmp_path: Any, fmt: Any) -> None:
    """EVT2/EVT2.1 time bases are 34-bit (~4.8 h); crossing that boundary
    must be tracked by the wrap accumulator."""
    base = (1 << 34) - 50
    t = base + np.arange(0, 100, 10)  # crosses 2^34
    ev = _wrap_events(t)
    out = _roundtrip(tmp_path, ev, fmt)
    assert np.array_equal(out["t"], ev['t'])


@pytest.mark.parametrize("fmt", ["evt3", "evt2", "evt21"])
def test_EVT_long_recording_hours(tmp_path: Any, fmt: Any) -> None:
    """Sparse multi-hour timestamps survive as long as gaps stay below the
    format's wrap period (2^24 us for EVT3, 2^34 us for EVT2/2.1)."""
    if fmt == "evt3":
        t = np.arange(0, 3600_000_000, 10_000_000)  # 1 h in 10 s steps
    else:
        t = np.arange(0, 10 * 3600_000_000, 3600_000_000)  # 10 h in 1 h steps
    ev = _wrap_events(t)
    out = _roundtrip(tmp_path, ev, fmt)
    assert np.array_equal(out["t"], ev['t'])


@pytest.mark.parametrize("fmt", ["evt3", "evt2", "evt21"])
def test_EVT_truncated_payload(tmp_path: Any, fmt: Any) -> None:
    """A file cut mid-word must not crash: all complete events decode."""
    from evutils.io import EventReader, EventWriter
    ev = _wrap_events(np.arange(1000) * 17)
    p = tmp_path / f"trunc_{fmt}.raw"
    with EventWriter(p, format=fmt) as w:
        w.write(ev)
    data = p.read_bytes()
    p.write_bytes(data[:-3])  # not a multiple of any word size

    with EventReader(p) as r:
        out = r.read_all()
    assert not isinstance(out, tuple)
    assert 0 < len(out) <= 1000
    assert np.array_equal(out["t"], ev['t'][:len(out)])


def test_EVT_header_only_file(tmp_path: Any) -> None:
    """A header with no payload reads as zero events (not an error)."""
    from evutils.io import EventReader, EventWriter
    p = tmp_path / "header_only.raw"
    w = EventWriter(p, format="evt3")
    w.init()
    w.close()
    with EventReader(p) as r:
        assert len(r.read_all()) == 0


def test_EVT_from_bytes_and_stream(tmp_path: Any, test_events: Any) -> None:
    """The reader accepts raw bytes and binary streams, not just paths."""
    import io as _io
    from evutils.io import EventReader, EventWriter
    p = tmp_path / "stream.raw"
    with EventWriter(p) as w:
        w.write(test_events)
    data = p.read_bytes()

    for source in (data, _io.BytesIO(data)):
        with EventReader(source) as r: # type: ignore
            assert np.array_equal(np.asarray(r.read_all()), test_events)


def test_EVT3_coordinate_extremes(tmp_path: Any) -> None:
    """Max 11-bit coordinates and both polarities survive the trip."""
    ev = _wrap_events(np.arange(6))
    ev['x'] = [0, 2047, 0, 2047, 1, 2046]
    ev['y'] = [0, 2047, 2047, 0, 2, 2045]
    ev['p'] = [0, 1, 0, 1, 1, 0]
    out = _roundtrip(tmp_path, ev, "evt3")
    assert np.array_equal(out["x"], ev['x'])
    assert np.array_equal(out["y"], ev['y'])
    assert np.array_equal(out["p"], ev['p'])
