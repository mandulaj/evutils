import os
import random
from pathlib import Path
import pytest
from evutils.io import EventReader, EventWriter
from evutils.types import Event_dtype
import numpy as np

FORMATS = [("evt3", "raw"), ("evt4", "raw"), ("evt2", "raw"), ("evt21", "raw"), ("dat", "dat"), ("csv", "csv")]

def _write_valid_file(path, fmt):
    ev = np.zeros(10, dtype=Event_dtype)
    ev["t"] = np.arange(10) * 1000
    if fmt in ("evt3", "evt4", "evt2", "evt21"):
        with EventWriter(path, format=fmt) as w:
            w.write(ev)
    else:
        with EventWriter(path, width=1280, height=720) as w:
            w.write(ev)

@pytest.mark.parametrize("fmt,ext", FORMATS)
def test_fuzz_parser_garbage(tmp_path, fmt, ext):
    """Feed random bytes after a valid header to ensure parsers don't segfault."""
    p = tmp_path / f"fuzz_{fmt}.{ext}"
    _write_valid_file(p, fmt)
    
    content = bytearray(p.read_bytes())
    
    # Append 1MB of random bytes
    random.seed(42)
    garbage = bytearray(random.getrandbits(8) for _ in range(100_000))
    content.extend(garbage)
    
    p.write_bytes(content)
    
    # Read it back. We expect it to either return garbage events, raise an error, or stop.
    # It must NOT segfault or hang indefinitely.
    with EventReader(p, mode="all") as r:
        try:
            r.read()
        except Exception:
            pass # Parsing errors are acceptable, segfaults are not
