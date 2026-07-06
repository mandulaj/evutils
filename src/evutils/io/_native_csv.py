from __future__ import annotations
import ctypes
from ctypes import POINTER, c_void_p, c_int, c_char_p, c_size_t, c_char
from ._native_core import register_bindings

def _bind_csv(handle: ctypes.CDLL) -> None:
    if hasattr(handle, "evutils_read_csv"):
        handle.evutils_read_csv.argtypes = [
            c_char_p,          # buffer
            c_size_t,          # buffer_len
            c_char,            # delimiter
            POINTER(c_void_p), # out_arrays
            POINTER(c_int),    # array_types
            POINTER(c_int),    # col_mapping
            c_int,             # max_csv_cols
            c_size_t,          # max_events
            POINTER(c_size_t), # bytes_consumed
            POINTER(c_size_t), # events_parsed
        ]
        handle.evutils_read_csv.restype = c_int

    if hasattr(handle, "evutils_write_csv"):
        handle.evutils_write_csv.argtypes = [
            POINTER(c_void_p), # in_arrays
            POINTER(c_int),    # array_types
            c_int,             # num_columns
            c_char,            # delimiter
            c_size_t,          # num_events
            c_char_p,          # out_buffer
            c_size_t,          # out_buffer_len
            POINTER(c_size_t), # bytes_written
            POINTER(c_size_t), # events_written
        ]
        handle.evutils_write_csv.restype = c_int

register_bindings(_bind_csv)
