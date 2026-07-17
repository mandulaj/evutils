"""ctypes bindings for the native CSV/TXT parser and writer.

Registers ``evutils_read_csv`` / ``evutils_write_csv`` (csrc/csv.c) with the
shared library handle; the Python-side chunking logic lives in
:mod:`evutils.io._csv`.
"""
from __future__ import annotations
import ctypes
from ctypes import POINTER, c_void_p, c_int, c_char_p, c_size_t, c_char, Structure, CDLL
from ._native_core import register_bindings

class ParserResult(Structure):
    _fields_ = [
        ("current", c_void_p),
        ("status", c_int)
    ]

def _bind_csv(handle: CDLL) -> None:
    if hasattr(handle, "evutils_read_csv"):
        handle.evutils_read_csv.argtypes = [
            c_char_p,          # const char *buffer
            c_size_t,          # size_t buffer_len
            c_char,            # char delimiter
            POINTER(c_void_p), # void **out_arrays
            POINTER(c_int),    # int *array_types
            POINTER(c_int),    # int *col_mapping
            c_int,             # int max_csv_cols
            c_size_t,          # size_t max_events
            POINTER(c_size_t), # size_t *events_parsed
        ]
        handle.evutils_read_csv.restype = ParserResult

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
