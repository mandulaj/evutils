import pytest
import numpy as np
from evutils.io import EventReader, EventWriter
from evutils.types import Event_dtype

def test_evt3_71_min_overflow():
    import numpy as np
    from evutils.io._native_evt import Evt3Parser, Evt3Input
    from evutils.io._native_core import EventSoABuffers, TriggerSoABuffers, parse_step

    # Create a sequence of wraps.
    # Each wrap is triggered when new_ts_high < local_state.ts_high.
    # So we write TIME_HIGH=0xFFF, then TIME_HIGH=0.
    # To reach 72 minutes (which is 257 wraps), we need 257 pairs of TIME_HIGH.
    
    words = []
    
    # 260 wraps = 260 * 16.7 seconds = 4342 seconds = 72.3 minutes
    for _ in range(260):
        words.append((0x8 << 12) | 0xFFF) # TIME_HIGH = 0xFFF
        words.append((0x8 << 12) | 0x000) # TIME_HIGH = 0x000
        
    # Now we are at wrap 260. ts_high_high = 260 * 0x1000000 = 0x104000000
    # Let's add a TIME_LOW = 0x123
    words.append((0x6 << 12) | 0x123)
    # Add an event at x=10, y=20, p=1
    words.append((0x0 << 12) | 20)
    words.append((0x2 << 12) | 10 | (1 << 11))
    
    # Pad with 4 words because EVT3 parser needs padding
    words.extend([0, 0, 0, 0])
    
    words = np.array(words, dtype=np.uint16)
    
    parser = Evt3Parser()
    ev = EventSoABuffers(100)
    ev.c.capacity = 100
    tr = TriggerSoABuffers(100)
    
    parse_step(words, 0, Evt3Input, parser, ev, tr, tail_pad=4, word_dtype=np.uint16)
    
    assert ev.size == 1
    # Expected timestamp: 0x104000000 (from 260 wraps) + 0 (from ts_high=0) + 0x123 (from ts_low)
    expected_ts = (260 * 0x1000000) + 0x123
    assert ev.t[0] == expected_ts
    assert ev.x[0] == 10
    assert ev.y[0] == 20
    assert ev.p[0] == 1
