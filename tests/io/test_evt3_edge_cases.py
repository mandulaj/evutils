"""EVT3 parser coverage gaps — recorded as skipped stubs to fill in later.

The normal recordings exercise the common EVT3 packets, but a few real branches
in ``EVT3_parse_chunk_soa`` (csrc/evt3.c) are never hit by them, and the
target-clone build makes branch coverage look worse than it is. Each test below
documents one gap: what input triggers the branch and what to assert. Drop the
``skip`` and implement the body when writing the real test.

These are correctness gaps, not known bugs.
"""
import pytest


@pytest.mark.skip(reason="TODO: exercise incomplete vector continuation branch")
def test_incomplete_vector_continuation() -> None:
    """A VECT_BASE_X / VECT_12 group whose follow-up word is NOT the expected
    continuation type.

    In ``EVT3_parse_vector_12_12_8_soa`` the two nested ``if
    (EVT3_get_packet_type(*current) == EVT3_VECT_12 / EVT3_VECT_8)`` checks
    (evt3.c ~225 / ~229) only ever take the "present" side with real data, so
    the "absent" side is never covered.

    How to test: hand-craft a uint16 EVT3 word stream (not a full recording) with
    a VECT_BASE_X followed by a single VECT_12 word and then a NON-VECT word
    (e.g. an EVT_ADDR_X), so the 2nd/3rd continuation is missing. Feed it through
    the low-level SOA parse step (see evutils.io._native_evt / _native_core
    parse_step) and assert the decoded event count matches only the bits present
    in the first vector word.
    """


@pytest.mark.skip(reason="TODO: exercise output-buffer-full early exit")
def test_output_buffer_full_stops_mid_chunk() -> None:
    """Main loop guard ``n_events_read < events_capacity_offset`` (and the
    trigger equivalent) tripping before the input is consumed.

    Normal runs size the output buffers generously, so the loop always exits on
    end-of-input, never on capacity — leaving the capacity guards at ~83%
    (evt3.c ~297 / ~322). This is the "small event array" case.

    How to test: drive ``EVT3_parse_chunk_soa`` with a small event buffer
    (capacity just above the 64-slot vector headroom) and enough input to
    overflow it. Assert the parser returns with ``current`` short of ``end`` and
    ``event_buffer.size`` at the cap, then that a second call resumes from
    ``current`` and drains the rest with no lost/duplicated events.
    """


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
