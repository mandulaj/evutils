import warnings

import numpy as np
import pytest






####################################
####################################
# Test CSV module
####################################
####################################





from typing import Any
def test_CSV_writer(tmp_path: Any, test_events: Any) -> None:
    from evutils.io import EventWriter
    from evutils.io import EventReader


    d = tmp_path / "sub"
    d.mkdir()
    p = d / "test.csv"
    writer = EventWriter(p)
    writer.write(test_events)
    writer.close()

    # Check if the file is created
    assert p.is_file()

    # load last line:
    with open(p, 'r') as f:
        lines = f.readlines()
        last_line = lines[-1]
        assert last_line == f"{test_events[-1]['t']},{test_events[-1]['x']},{test_events[-1]['y']},{test_events[-1]['p']}\n"

        assert len(lines) == len(test_events) + 1


    reader = EventReader(p, n_events=len(test_events))
    events = reader.read()
    assert np.array_equal(events, test_events)

def test_CSV_reader_nevents(event_files: Any, test_events: Any) -> None:
    csv_file_path = event_files['csv']

    from evutils.io import EventReader

    STEP=100

    with EventReader(csv_file_path, n_events=STEP) as reader:

        for i in range(0, len(test_events), STEP):
            events = reader.read()
            assert np.array_equal(events, test_events[i:i+STEP])


def test_CSV_reader_gen(event_files: Any, test_events: Any) -> None:
    csv_file_path = event_files['csv']

    from evutils.io import EventReader

    STEP=100
    i = 0

    for ev in EventReader(csv_file_path, n_events=STEP):

        assert np.array_equal(ev, test_events[i:i+STEP])
        i += STEP


####################################
# Header / column-order handling
####################################

def _small_events(n: int = 50) -> Any:
    from evutils.types import Event_dtype
    ev = np.zeros(n, dtype=Event_dtype)
    ev['t'] = np.arange(n, dtype=np.int64) * 10
    ev['x'] = np.arange(n) % 1280
    ev['y'] = np.arange(n) % 720
    ev['p'] = np.arange(n) % 2
    return ev


def test_CSV_headerless_default_order(tmp_path: Any) -> None:
    """No header, no order parameter: the default [t, x, y, p] is assumed."""
    from evutils.io import EventReader, EventWriter
    ev = _small_events()
    p = tmp_path / "noheader.csv"
    with EventWriter(p, header=False) as w:
        w.write(ev)
    assert not open(p).readline().startswith("t")  # really headerless

    with EventReader(p, mode="all") as r:
        assert np.array_equal(np.asarray(r.read()), ev)


def test_CSV_headerless_custom_order(tmp_path: Any) -> None:
    """Write and read back with a shuffled column order (no header)."""
    from evutils.io import EventReader, EventWriter
    ev = _small_events()
    order = ['x', 'y', 'p', 't']
    p = tmp_path / "xypt.csv"
    with EventWriter(p, header=False, order=order) as w:
        w.write(ev)

    with EventReader(p, mode="all", order=order) as r:
        assert np.array_equal(np.asarray(r.read()), ev)


def test_CSV_header_with_custom_order(tmp_path: Any) -> None:
    """A shuffled header alone must drive the column mapping on read."""
    from evutils.io import EventReader, EventWriter
    ev = _small_events()
    p = tmp_path / "hdr_xypt.csv"
    with EventWriter(p, header=True, order=['x', 'y', 'p', 't']) as w:
        w.write(ev)
    assert open(p).readline().strip() == "x,y,p,t"

    with EventReader(p, mode="all") as r:  # order inferred from header
        assert np.array_equal(np.asarray(r.read()), ev)


def test_CSV_header_overrides_order_param_with_warning(tmp_path: Any) -> None:
    """When the file header disagrees with the order parameter, the header
    wins and a warning is emitted."""
    import warnings
    from evutils.io import EventReader, EventWriter
    ev = _small_events()
    p = tmp_path / "conflict.csv"
    with EventWriter(p, header=True) as w:  # header: t,x,y,p
        w.write(ev)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with EventReader(p, mode="all", order=['x', 'y', 'p', 't']) as r:
            out = r.read()
    assert any("precedence" in str(w.message) for w in caught)
    assert np.array_equal(np.asarray(out), ev)


def test_CSV_invalid_order_rejected(tmp_path: Any) -> None:
    from evutils.io import EventReader
    p = tmp_path / "dummy.csv"
    p.write_text("t,x,y,p\n0,1,2,1\n")
    with pytest.raises(ValueError):
        EventReader(p, order=['t', 'x', 'y'])  # not 4 entries
    with pytest.raises(ValueError):
        EventReader(p, order=['t', 'x', 'y', 'z'])  # missing 'p'


####################################
# Delimiters
####################################

@pytest.mark.parametrize("delim", [";", "\t", " "])
def test_CSV_delimiter_roundtrip(tmp_path: Any, delim: Any) -> None:
    from evutils.io import EventReader, EventWriter
    ev = _small_events()
    p = tmp_path / "delim.csv"
    with EventWriter(p, sep=delim) as w:
        w.write(ev)
    assert delim in open(p).readline()

    with EventReader(p, mode="all", delimiter=delim) as r:
        assert np.array_equal(np.asarray(r.read()), ev)


####################################
# Malformed / edge-case inputs
####################################

def test_CSV_empty_file(tmp_path: Any) -> None:
    from evutils.io import EventReader
    p = tmp_path / "empty.csv"
    p.touch()
    with EventReader(p, mode="all") as r:
        assert len(r.read()) == 0


def test_CSV_header_only_file(tmp_path: Any) -> None:
    from evutils.io import EventReader
    p = tmp_path / "hdr_only.csv"
    p.write_text("t,x,y,p\n")
    with EventReader(p, mode="all") as r:
        assert len(r.read()) == 0


def test_CSV_missing_trailing_newline(tmp_path: Any) -> None:
    """The final line must be parsed even without a terminating newline."""
    from evutils.io import EventReader
    p = tmp_path / "notrail.csv"
    p.write_text("t,x,y,p\n10,1,2,1\n20,3,4,0")  # no final \n
    with EventReader(p, mode="all") as r:
        out = r.read()
    assert len(out) == 2
    assert out['t'][-1] == 20 and out['x'][-1] == 3 and out['p'][-1] == 0


def test_CSV_blank_lines_skipped(tmp_path: Any) -> None:
    from evutils.io import EventReader
    p = tmp_path / "blanks.csv"
    p.write_text("t,x,y,p\n10,1,2,1\n\n\n20,3,4,0\n\n")
    with EventReader(p, mode="all") as r:
        out = r.read()
    assert np.array_equal(out['t'], [10, 20])


def test_CSV_crlf_line_endings(tmp_path: Any) -> None:
    from evutils.io import EventReader
    p = tmp_path / "crlf.csv"
    p.write_bytes(b"t,x,y,p\r\n10,1,2,1\r\n20,3,4,0\r\n")
    with EventReader(p, mode="all") as r:
        out = r.read()
    assert np.array_equal(out['t'], [10, 20])
    assert np.array_equal(out['p'], [1, 0])


def test_CSV_non_numeric_field_parses_as_zero(tmp_path: Any) -> None:
    """Documented C-parser semantics: garbage fields decode to 0 (no crash,
    no exception, no line skip)."""
    from evutils.io import EventReader
    p = tmp_path / "garbage.csv"
    p.write_text("t,x,y,p\n10,abc,2,1\n20,3,4,0\n")
    with EventReader(p, mode="all") as r:
        out = r.read()
    assert np.array_equal(out['t'], [10, 20])
    assert out['x'][0] == 0  # 'abc' -> 0
    assert out['x'][1] == 3


def test_CSV_extra_columns_ignored(tmp_path: Any) -> None:
    from evutils.io import EventReader
    p = tmp_path / "extra.csv"
    p.write_text("t,x,y,p\n10,1,2,1,999,777\n20,3,4,0,888\n")
    with EventReader(p, mode="all") as r:
        out = r.read()
    assert np.array_equal(out['t'], [10, 20])
    assert np.array_equal(out['x'], [1, 3])


def test_CSV_short_line_does_not_crash(tmp_path: Any) -> None:
    """A line with missing fields must not crash or derail later lines.

    (Values for the absent fields are unspecified; only stability and the
    surrounding rows are asserted.)"""
    from evutils.io import EventReader
    p = tmp_path / "short.csv"
    p.write_text("t,x,y,p\n10,1,2,1\n20,3\n30,5,6,0\n")
    with EventReader(p, mode="all") as r:
        out = r.read()
    assert len(out) == 3
    assert out['t'][0] == 10 and out['t'][2] == 30
    assert out['x'][2] == 5 and out['p'][2] == 0


def test_CSV_short_line_warns(tmp_path: Any) -> None:
    """A short row (missing a mapped column) surfaces a malformed-row warning,
    matching the binary parsers' malformed-packet warning."""
    from evutils.io import EventReader
    p = tmp_path / "short_warn.csv"
    p.write_text("t,x,y,p\n10,1,2,1\n20,3\n30,5,6,0\n")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with EventReader(p, mode="all") as r:
            out = r.read()
    assert len(out) == 3
    msgs = [str(w.message) for w in caught]
    assert any("malformed" in m.lower() and "CSV row" in m for m in msgs), msgs


def test_CSV_short_line_strict_raises(tmp_path: Any) -> None:
    """strict=True turns a malformed CSV row into an error, like the binary
    formats."""
    from evutils.io import EventReader
    p = tmp_path / "short_strict.csv"
    p.write_text("t,x,y,p\n10,1,2,1\n20,3\n30,5,6,0\n")
    with pytest.raises(RuntimeError, match="malformed"):
        with EventReader(p, mode="all", strict=True) as r:
            r.read()


def test_CSV_wellformed_no_warning(tmp_path: Any) -> None:
    """Clean rows (incl. extra trailing columns) emit no malformed warning."""
    from evutils.io import EventReader
    p = tmp_path / "clean.csv"
    p.write_text("t,x,y,p\n10,1,2,1,999\n20,3,4,0\n")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with EventReader(p, mode="all") as r:
            r.read()
    assert not any("malformed" in str(w.message).lower() for w in caught)


def test_CSV_negative_and_extreme_values(tmp_path: Any) -> None:
    """Signed 64-bit timestamps and full uint16 coordinates round-trip."""
    from evutils.io import EventReader, EventWriter
    from evutils.types import Event_dtype
    ev = np.zeros(4, dtype=Event_dtype)
    ev['t'] = [-1_000_000, 0, 2**62, 2**62 + 1]
    ev['x'] = [0, 65535, 1, 2]
    ev['y'] = [65535, 0, 3, 4]
    ev['p'] = [0, 1, 1, 0]
    p = tmp_path / "extreme.csv"
    with EventWriter(p) as w:
        w.write(ev)
    with EventReader(p, mode="all") as r:
        assert np.array_equal(np.asarray(r.read()), ev)


def test_CSV_whitespace_padding(tmp_path: Any) -> None:
    """Fields padded with spaces/tabs still parse."""
    from evutils.io import EventReader
    p = tmp_path / "padded.csv"
    p.write_text("t,x,y,p\n 10 , 1 ,2, 1\n\t20,3,\t4,0\n")
    with EventReader(p, mode="all") as r:
        out = r.read()
    assert np.array_equal(out['t'], [10, 20])
    assert np.array_equal(out['x'], [1, 3])
    assert np.array_equal(out['p'], [1, 0])


def test_CSV_chunked_reads_across_refills(tmp_path: Any) -> None:
    """Events spanning multiple internal 4 MB refills stay contiguous and
    ordered (exercises the buffered consume/refill loop)."""
    from evutils.io import EventReader, EventWriter
    from evutils.types import Event_dtype
    n = 200_000  # ~2.5 MB of text; chunk_size below forces many parse calls
    ev = np.zeros(n, dtype=Event_dtype)
    ev['t'] = np.arange(n, dtype=np.int64)
    ev['x'] = np.arange(n) % 1280
    ev['y'] = np.arange(n) % 720
    ev['p'] = np.arange(n) % 2
    p = tmp_path / "big.csv"
    with EventWriter(p) as w:
        w.write(ev)

    total = 0
    with EventReader(p, n_events=7_777) as r:
        for chunk in r:
            assert np.array_equal(chunk['t'], ev['t'][total:total + len(chunk)])
            total += len(chunk)
    assert total == n

