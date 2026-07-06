"""Write-throughput benchmarks for the EVT2 / EVT2.1 / EVT3 encoders.

Run with::

    pytest benchmarks/test_write.py

The payload is the shared ``reference_events`` array, so all three formats
encode the same events. ``warmup_rounds=1`` absorbs the numba JIT compile of
each encoder so it isn't counted in the timed rounds.
"""
import pytest

from typing import Any

from evutils.io import EventWriter  # type: ignore

FORMATS = ["evt3", "evt2", "evt21"]


@pytest.mark.parametrize("fmt", FORMATS)
def test_write_evutils(benchmark: Any, benchmark_rounds: int, reference_events: Any, tmp_path: Any, fmt: str) -> None:
    benchmark.group = f"write-{fmt}"
    events = reference_events
    out = tmp_path / f"out_{fmt}.raw"

    def write() -> None:
        with EventWriter(out, format=fmt) as writer:
            writer.write(events)

    benchmark.pedantic(write, rounds=benchmark_rounds, iterations=1, warmup_rounds=1)

    assert out.stat().st_size > 0
    from evutils.io import EventReader
    assert len(EventReader(out).read_all()) == len(events)
    benchmark.extra_info["n_events"] = len(events)
    benchmark.extra_info["library"] = "evutils"


@pytest.mark.parametrize("fmt", ["evt2", "evt3"])
def test_write_expelliarmus(benchmark: Any, benchmark_rounds: int, reference_events: Any, tmp_path: Any, fmt: str) -> None:
    benchmark.group = f"write-{fmt}"
    try:
        from expelliarmus import Wizard  # type: ignore
    except ImportError as exc:
        pytest.skip(f"expelliarmus not available: {exc}")
    
    events = reference_events
    out = tmp_path / f"out_{fmt}.raw"
    wizard = Wizard(encoding=fmt)

    def write() -> None:
        wizard.save(str(out), events)

    benchmark.pedantic(write, rounds=benchmark_rounds, iterations=1, warmup_rounds=1)

    assert out.stat().st_size > 0
    assert len(wizard.read(str(out))) == len(events)
    benchmark.extra_info["n_events"] = len(events)
    benchmark.extra_info["library"] = "expelliarmus"
