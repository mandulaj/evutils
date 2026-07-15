# Legacy benchmark suite (superseded)

These files are the old **pytest-benchmark** suite. They are **superseded by
`benchmarks/throughput.py`** (a single in-RAM driver that reports throughput in
M events/s as two matrices) and are kept here only for reference. They are **not
maintained** and are not part of any CI run.

Why they were retired:
- reported wall-clock time (not scale-invariant across recordings of different
  sizes),
- read from disk rather than a RAM disk,
- still exercised `read_all()`, which is useless for real recordings that don't
  fit in memory,
- used a synthetic 5M-event write payload instead of real data.

If you need the old behaviour: `pytest benchmarks/legacy/`.
`generate_benchmark_table.py` parsed the old pytest-benchmark JSON; the new
driver emits markdown directly via `throughput.py --markdown`.
