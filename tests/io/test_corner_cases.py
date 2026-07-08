import numpy as np
import pytest
from evutils.io import EventReader, EventWriter
from evutils.io._source import StreamSource
import io

def test_trigger_boundary_sync(tmp_path):
    # Triggers and events at exactly the window boundary should be in the same chunk
    p = tmp_path / "boundary.raw"
    
    # 2 events: t=0 and t=1000
    ev = np.zeros(2, dtype=np.dtype([('t', np.int64), ('x', np.uint16), ('y', np.uint16), ('p', np.uint8)]))
    ev['t'] = [0, 1000]
    
    with EventWriter(p, format="evt3") as w:
        w.write(ev)
        
    # Append a trigger at t=1000
    # EVT3 trigger: TIME_HIGH, TIME_LOW, TRIGGER
    t = 1000
    words = np.array([
        0x8000 | ((t >> 12) & 0xFFF),
        0x6000 | (t & 0xFFF),
        0xA000 | ((1 & 0xF) << 8) | 1,
    ], dtype=np.uint16)
    with open(p, "ab") as f:
        f.write(words.tobytes())

    with EventReader(p, ext_trigger=True, mode="delta_t", delta_t=1000) as r:
        chunks = list(r)
        
    # Chunk 1: [0, 1000)
    ev1, tr1 = chunks[0]
    assert len(ev1) == 1 and ev1.t[0] == 0
    assert len(tr1) == 0  # Trigger at 1000 should NOT be here
    
    # Chunk 2: [1000, 2000)
    ev2, tr2 = chunks[1]
    assert len(ev2) == 1 and ev2.t[0] == 1000
    assert len(tr2) == 1 and tr2.t[0] == 1000 # Trigger at 1000 should be here




def test_extreme_playback_speed(tmp_path):
    p = tmp_path / "extreme.raw"
    ev = np.zeros(2, dtype=np.dtype([('t', np.int64), ('x', np.uint16), ('y', np.uint16), ('p', np.uint8)]))
    ev['t'] = [0, 1_000_000] # 1 second apart
    with EventWriter(p, format="evt3") as w:
        w.write(ev)
        
    # extremely fast playback speed should not sleep
    import time
    start = time.perf_counter()
    with EventReader(p, mode="all", real_time=True, playback_speed=1e6) as r:
        out = r.read()
    elapsed = time.perf_counter() - start
    assert elapsed < 0.5  # Should take well under 1 second
