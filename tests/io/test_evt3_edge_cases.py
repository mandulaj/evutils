"""EVT3 parser coverage gaps — recorded as skipped stubs to fill in later.

The normal recordings exercise the common EVT3 packets, but a few real branches
in ``EVT3_parse_chunk_soa`` (csrc/evt3.c) are never hit by them, and the
target-clone build makes branch coverage look worse than it is. Each test below
documents one gap: what input triggers the branch and what to assert. Drop the
``skip`` and implement the body when writing the real test.

These are correctness gaps, not known bugs.
"""
import pytest


def test_incomplete_vector_continuation() -> None:
    import numpy as np
    from evutils.io._native_evt import Evt3Parser, Evt3Input
    from evutils.io._native_core import EventSoABuffers, TriggerSoABuffers, parse_step

    # VECT_BASE_X (type 3) followed by the FIRST VECT_12 (type 4).
    # Normally a second VECT_12 follows. Here, we interrupt with EVT_ADDR_X (type 2).
    words = np.array([
        (0x3 << 12) | 0x0,
        (0x4 << 12) | 0xFFF,
        (0x2 << 12) | 200,
    ], dtype=np.uint16)

    parser = Evt3Parser()
    ev = EventSoABuffers(100)
    ev.c.capacity = 100
    tr = TriggerSoABuffers(100)

    parse_step(words, 0, Evt3Input, parser, ev, tr, tail_pad=4, word_dtype=np.uint16)

    # 12 events from the first vector word + 1 from the EVT_ADDR_X word = 13 events.
    # The interruption aborts the continuation gracefully.
    assert ev.size == 13

def test_output_buffer_full_stops_mid_chunk() -> None:
    import numpy as np
    from evutils.io._native_evt import Evt3Parser, Evt3Input
    from evutils.io._native_core import EventSoABuffers, TriggerSoABuffers, parse_step

    # VECT_BASE_X + 6 full blocks of (VECT_12, VECT_12, VECT_8).
    # Each full block emits 32 events. 6 blocks = 192 events.
    words = [(0x3 << 12) | 0x0]
    block = [
        (0x4 << 12) | 0xFFF,
        (0x4 << 12) | 0xFFF,
        (0x5 << 12) | 0xFF,
    ]
    words.extend(block * 6)
    words = np.array(words, dtype=np.uint16)

    parser = Evt3Parser()
    # Create an event buffer with a capacity of 100.
    # events_capacity_offset is 100 - 64 = 36.
    ev = EventSoABuffers(100)
    ev.c.capacity = 100
    tr = TriggerSoABuffers(100)

    # First parse: should stop after 2 blocks (64 events) because 64 >= 36.
    appended, off = parse_step(words, 0, Evt3Input, parser, ev, tr, tail_pad=4, word_dtype=np.uint16)
    assert appended == 64
    assert off == 7 # 1 VECT_BASE_X + 2 blocks * 3 words = 7 words consumed
    assert ev.size == 64

    # "Drain" the buffer and resume from `off`
    ev.reset()
    appended2, off2 = parse_step(words, off, Evt3Input, parser, ev, tr, tail_pad=4, word_dtype=np.uint16)
    assert appended2 == 64 # 2 more blocks
    assert off2 == 13 # 7 + 2 * 3 = 13
    assert ev.size == 64


@pytest.mark.skip(reason="Measurement note, not a runtime test — see docstring")
def test_note_target_clone_branch_coverage() -> None:
    """gcovr under-reports EVT3 branch coverage because of function multi-versioning.

    ``EVUTILS_TARGET_CLONES`` (csrc/include/evutils/compat.h) tags the hot chunk
    parsers with ``__attribute__((target_clones("avx2","default")))`` on
    GCC/x86-64, so the compiler emits TWO instantiations of each function. The
    coverage host runs only one clone (whichever its CPU dispatches to); gcovr
    still counts the branches of BOTH, so the unrun clone shows every branch as
    "not taken" and halves the reported branch-rate.

    Fix for a later PR (not code to run here): build the coverage target with the
    clones disabled so branches attribute to a single instantiation. E.g. give
    ``EVUTILS_COVERAGE`` in CMakeLists.txt a ``-DEVUTILS_NO_TARGET_CLONES``
    define and make ``EVUTILS_TARGET_CLONES`` expand to nothing when it is set.
    Then re-run scripts/coverage.sh and confirm evt3.c branch-rate jumps.
    """
