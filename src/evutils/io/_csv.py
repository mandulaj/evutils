"""CSV file decoder and encoder."""

import io
import warnings
from datetime import datetime
from io import TextIOWrapper
from pathlib import Path

import numpy as np
from ..types import Event_dtype, EventArray, TriggerArray
from .common import EventDecoder, EventEncoder

class EventDecoder_Csv(EventDecoder):
    """A reader for CSV files with events.

    Parameters
    ----------
    file
        Path to the data file
    order
        Order of the columns in the CSV file, by default ['t', 'x', 'y', 'p']
    chunk_size
        Size of the chunk to read, by default 10_000
    delimiter
        Delimiter for the CSV file, by default ","
    engine
        Engine to use to read the CSV file, by default 'c'

    Raises
    ------
    ValueError
        If the order is not a list of 4 strings or if the order does not contain 't', 'x', 'y' and 'p'

    """

    #: read_chunk parses into fresh, independent arrays bounded by n_events_hint,
    #: so EventReader can hand them out directly (skipping the staging
    #: accumulator). CSV decode is text-parse-bound, so the gain is small.
    _independent_windows = True

    #: Seekable via byte-offset binary search with newline resync (time) or a
    #: newline count (event index). Requires a seekable source.
    SUPPORTS_SEEK = True

    def __init__(self, readable:io.BufferedReader,  order:list[str]|None=None, chunk_size:int=1_000_000, delimiter:str=",", engine:str='c'):
        super().__init__(readable, chunk_size)

        if order is None:
            # we infer the header from the file, or use a default header
            pass
        else:
            # Validate the parameters
            if len(order) != 4:
                raise ValueError("Order must be a list of 4 strings")
            if "t" not in order or "x" not in order or "y" not in order or "p" not in order:
                raise ValueError("Order must contain 't', 'x', 'y' and 'p'")

        self._order = order
        self._delimiter = delimiter
        self._engine = engine

    def _check_header(self) -> None:
        have_header = False

        # Check first line to see if it is a header
        # TODO: Need to deal with non-seekable files
        first_line: str = self._fd.readline().decode('utf-8').strip()

        # Check if the first line is a header
        # If we find a header, it will take precendence over the order parameter

        cols = [c.strip() for c in first_line.split(self._delimiter)]
        if "t" in cols or "x" in cols or "y" in cols or "p" in cols:
            # Header found
            have_header = True

        if have_header:
            # We expect a header:
            if "t" in cols and "x" in cols and "y" in cols and "p" in cols:
                # Header found
                order = cols

                if self._order is not None and order != self._order:
                    warnings.warn(f"Header order {order} in file takes precedence over requested order {self._order}")
                self._order = order

            else:
                raise ValueError(f"Header not found or invalid: {first_line}")
        else:
            # No header found, just seek to start of file
            self._fd.seek(0)

        # If we still don't have a header, we use a default one
        if self._order is None:
            self._order = ['t', 'x', 'y', 'p']

    def init(self) -> None:
        """Initialize the CSV reader.

        Returns
        -------
        None

        """
        self._check_header()
        assert self._order is not None  # set by _check_header

        # Build col_mapping: maps CSV index to out_array index
        # out_arrays index: 0=t, 1=x, 2=y, 3=p
        self._field_map = {'t': 0, 'x': 1, 'y': 2, 'p': 3}
        self._col_mapping = [-1] * len(self._order)
        for i, col in enumerate(self._order):
            if col in self._field_map:
                self._col_mapping[i] = self._field_map[col]

        # Byte offset of the first event line (past any header), for seeking.
        self._data_start = int(self._fd.tell())

        self._buffer = bytearray()
        self._is_initialized = True

    def read_chunk(self, delta_t_hint:int | None = None, n_events_hint:int | None = None) -> 'EventArray':
        """Read a chunk of events from the CSV file."""
        import ctypes
        from . import _native_csv; from ._native_core import lib
        
        assert self._is_initialized, "Reader is not initialized"
        chunk_size = self._chunk_size
        if n_events_hint is not None:
            chunk_size = n_events_hint

        t_arr = np.zeros(chunk_size, dtype=np.int64)
        x_arr = np.zeros(chunk_size, dtype=np.uint16)
        y_arr = np.zeros(chunk_size, dtype=np.uint16)
        p_arr = np.zeros(chunk_size, dtype=np.uint8)

        array_types = (ctypes.c_int * 4)(8, 2, 2, 1)
        col_mapping = (ctypes.c_int * len(self._col_mapping))(*self._col_mapping)
        delimiter = self._delimiter.encode('utf-8')[0]

        events_parsed_total = 0

        while events_parsed_total < chunk_size:
            if not self._eof and len(self._buffer) < 1024 * 1024:
                new_data = self._fd.read(4 * 1024 * 1024)
                if not new_data:
                    self._eof = True
                    self._buffer.extend(b'\n') # Guarantee last line ends with newline
                else:
                    self._buffer.extend(new_data)

            if len(self._buffer) == 0:
                break

            bytes_consumed = ctypes.c_size_t(0)
            events_parsed = ctypes.c_size_t(0)

            cur_out_ptrs = (ctypes.c_void_p * 4)(
                t_arr.ctypes.data + events_parsed_total * 8,
                x_arr.ctypes.data + events_parsed_total * 2,
                y_arr.ctypes.data + events_parsed_total * 2,
                p_arr.ctypes.data + events_parsed_total * 1
            )

            c_buf = (ctypes.c_char * len(self._buffer)).from_buffer(self._buffer)

            lib().evutils_read_csv(
                c_buf, len(self._buffer), delimiter, cur_out_ptrs, array_types,
                col_mapping, len(self._col_mapping), chunk_size - events_parsed_total,
                ctypes.byref(bytes_consumed), ctypes.byref(events_parsed)
            )
            
            del c_buf # Release memory view so buffer can be resized

            consumed = bytes_consumed.value
            parsed = events_parsed.value

            if consumed == 0:
                if self._eof:
                    break
                # No full line in the buffered window (a line longer than the
                # refill threshold): pull more data regardless of the usual
                # low-water mark, otherwise this loop would never progress.
                new_data = self._fd.read(4 * 1024 * 1024)
                if not new_data:
                    self._eof = True
                    self._buffer.extend(b'\n')  # guarantee final-line newline
                else:
                    self._buffer.extend(new_data)
                continue

            del self._buffer[:consumed]
            events_parsed_total += parsed

        if events_parsed_total < chunk_size:
            self._eof = True

        return EventArray(
            t_arr[:events_parsed_total],
            x_arr[:events_parsed_total],
            y_arr[:events_parsed_total],
            p_arr[:events_parsed_total]
        )

    # ------------------------------------------------------------------ #
    # Seeking (byte-offset binary search + newline resync)
    # ------------------------------------------------------------------ #
    def _file_size(self) -> int:
        cur = self._fd.tell()
        end = self._fd.seek(0, io.SEEK_END)
        self._fd.seek(cur)
        return end

    def _line_start_at_or_after(self, pos: int) -> int:
        """Byte offset of the first line whose start is ``>= pos``."""
        if pos <= self._data_start:
            return self._data_start
        self._fd.seek(pos - 1)  # find the newline at/after pos-1; line starts after it
        acc = 0
        while True:
            chunk = self._fd.read(65536)
            if not chunk:
                return pos - 1 + acc  # EOF, no newline
            i = chunk.find(b"\n")
            if i >= 0:
                return pos - 1 + acc + i + 1
            acc += len(chunk)

    def _parse_line_t(self, pos: int) -> int | None:
        """Parse the ``t`` column of the line starting at ``pos`` (or ``None``)."""
        assert self._order is not None
        self._fd.seek(pos)
        line = b""
        while True:
            chunk = self._fd.read(65536)
            if not chunk:
                break
            nl = chunk.find(b"\n")
            if nl >= 0:
                line += chunk[:nl]
                break
            line += chunk
        if not line.strip():
            return None
        parts = line.split(self._delimiter.encode("utf-8"))
        ti = self._order.index("t")
        try:
            return int(parts[ti].strip())
        except (ValueError, IndexError):
            return None

    def _seek_line_index(self, n: int) -> int:
        """Byte offset of event line ``n`` (0-based), by counting newlines."""
        self._fd.seek(self._data_start)
        pos = self._data_start
        remaining = n
        while remaining > 0:
            chunk = self._fd.read(1 << 20)
            if not chunk:
                break
            cnt = chunk.count(b"\n")
            if cnt < remaining:
                remaining -= cnt
                pos += len(chunk)
            else:
                idx = -1
                for _ in range(remaining):
                    idx = chunk.find(b"\n", idx + 1)
                pos += idx + 1
                remaining = 0
        return pos

    def seek(self, t: int | None = None, n: int | None = None) -> int:
        """Seek to an absolute timestamp (µs) or event index. See base class."""
        if not self._is_initialized:
            self.init()
        axis, val = self._seek_axis(t, n)
        size = self._file_size()

        if axis == "n":
            pos = self._seek_line_index(val)
        else:
            lo, hi = self._data_start, size
            while lo < hi:
                mid = (lo + hi) // 2
                ls = self._line_start_at_or_after(mid)
                tv = self._parse_line_t(ls) if ls < size else None
                if tv is not None and tv >= val:
                    hi = mid
                else:
                    lo = mid + 1
            pos = self._line_start_at_or_after(lo)

        self._fd.seek(pos)
        self._buffer = bytearray()
        self._eof = False
        landed = self._parse_line_t(pos) if pos < size else None
        self._fd.seek(pos)  # _parse_line_t moved the cursor
        return landed if landed is not None else val

    def reset(self) -> None:
        """Reset the CSV reader to the beginning of the file."""
        assert self._fd is not None
        self._fd.seek(0)
        self._eof = False
        if self._is_initialized:
            self._is_initialized = False
            self.init()

class EventEncoder_Csv(EventEncoder):
    """A writer for CSV files with events.

    Parameters
    ----------
    file : str
        Path to the data file
    width : int, optional
        Width of the frame, by default 1280 (not relevant for this formats)
    height : int, optional
        Height of the frame, by default 720 (not relevant for this formats)
    sep : str, optional
        Separator for the CSV file, by default ","
    order : list, optional
        Order of the columns in the CSV file, by default ['t', 'x', 'y', 'p']
    header : bool, optional
        If True, a header is written on the first line, by default True

    Raises
    ------
    ValueError
        If the order is not a list of 4 strings or if the order does not contain 't', 'x', 'y' and 'p'

    """

    def __init__(self, writable: io.BufferedWriter, width:int=1280, height:int=720, dt:datetime|None=None, sep:str=",", order:list[str]|None=None, header:bool=True):
        super().__init__(writable, width, height, dt)
        if order is None:
            order = ['t', 'x', 'y', 'p']

        if len(order) != 4:
            raise ValueError("Order must be a list of 4 strings")
        if "t" not in order or "x" not in order or "y" not in order or "p" not in order:
            raise ValueError("Order must contain 't', 'x', 'y' and 'p'")

        self._order = order
        self._header = header
        self._sep = sep

    def init(self) -> None:
        """Initialize the CSV writer.

        Returns
        -------
        None

        """
        if self._header:
            header = self._sep.join(self._order) + "\n"
            self._fd.write(header.encode('utf-8'))

        self._is_initialized = True

    def write(self, events: 'np.ndarray | EventArray', triggers: 'np.ndarray | TriggerArray | None' = None) -> int:
        """Write events to the CSV file."""
        import ctypes
        from . import _native_csv; from ._native_core import lib
        
        if not self._is_initialized:
            self.init()

        if isinstance(events, np.ndarray):
            events = EventArray.from_aos(events)

        chunk_size = len(events)
        if chunk_size == 0:
            return 0

        t_arr = events.t
        x_arr = events.x
        y_arr = events.y
        p_arr = events.p

        in_ptrs_list = []
        array_types_list = []
        for col in self._order:
            if col == 't':
                in_ptrs_list.append(t_arr.ctypes.data)
                array_types_list.append(8)
            elif col == 'x':
                in_ptrs_list.append(x_arr.ctypes.data)
                array_types_list.append(2)
            elif col == 'y':
                in_ptrs_list.append(y_arr.ctypes.data)
                array_types_list.append(2)
            elif col == 'p':
                in_ptrs_list.append(p_arr.ctypes.data)
                array_types_list.append(1)
            else:
                in_ptrs_list.append(0)
                array_types_list.append(0)

        in_ptrs = (ctypes.c_void_p * len(self._order))(*in_ptrs_list)
        array_types = (ctypes.c_int * len(self._order))(*array_types_list)
        delimiter = self._sep.encode('utf-8')[0]

        out_buffer_len = chunk_size * len(self._order) * 22
        out_buffer = ctypes.create_string_buffer(out_buffer_len)

        bytes_written = ctypes.c_size_t(0)
        events_written = ctypes.c_size_t(0)

        lib().evutils_write_csv(
            in_ptrs, array_types, len(self._order), delimiter, chunk_size,
            out_buffer, out_buffer_len, ctypes.byref(bytes_written), ctypes.byref(events_written)
        )

        self._fd.write(out_buffer.raw[:bytes_written.value])
        return events_written.value
