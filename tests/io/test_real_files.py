"""Format-agnostic checks against the real downloaded recordings.

Every file in the reference tarball ships a JSON sidecar (produced by
``scripts/generate_metadata.py``) carrying the ground-truth ``format``,
``resolution``, event/polarity/trigger counts, etc. This module discovers
*all* of them -- whatever the format -- and verifies the decoder output
against that ground truth.

Adding a new format is zero-code: drop the recording plus its JSON sidecar
into the tarball and it is exercised here automatically. The per-file
parametrization (fixture ``real_event_file``) is generated in
``tests/conftest.py::pytest_generate_tests``.
"""
import numpy as np
import pytest

from typing import Any

# Recordings whose Metavision reference counts differ slightly from a
# byte-exact decode: Metavision filters a handful of events, while our decoder
# matches expelliarmus exactly. Allow a tiny relative tolerance on *event*
# counts for these (triggers are always exact). See tests/io/test_evt.py for
# the investigation behind each number.
_COUNT_TOLERANT_FILES = {
    'evt2_195_falling_particles.raw',   # +14
    'evt3_flash.raw',                   # +3345
    'evt3_laser.raw',                   # +132
}


def test_real_file_matches_metadata(real_event_file: Any) -> None:
    """Decode a real recording end-to-end and check it against its JSON sidecar.

    Verifies (only what the metadata provides, so it stays format-agnostic):
    sensor resolution, coordinate bounds, total/positive/negative event counts,
    external-trigger counts, and an overall upward timestamp trend.
    """
    from evutils.io import EventReader
    from evutils.io.decoders import get_reader_from_filename

    ef = real_event_file
    meta = ef.metadata

    # Only ask for triggers from formats whose decoder can actually read them;
    # for the rest we instead assert the metadata isn't hiding any (below).
    supports_triggers = getattr(
        get_reader_from_filename(ef.path), "SUPPORTS_EXT_TRIGGERS", False
    )

    n = n_pos = n_neg = 0
    n_tr_pos = n_tr_neg = 0
    x_max = y_max = -1
    first_t = last_t = None

    with EventReader(ef.path, ext_trigger=supports_triggers, chunk_size=10_000_000) as reader:
        shape = list(reader.shape())
        for chunk in reader:
            ev, tr = chunk if isinstance(chunk, tuple) else (chunk, None)
            if len(ev):
                n += len(ev)
                n_pos += int(np.count_nonzero(ev["p"] == 1))
                n_neg += int(np.count_nonzero(ev["p"] == 0))
                x_max = max(x_max, int(ev["x"].max()))
                y_max = max(y_max, int(ev["y"].max()))
                if first_t is None:
                    first_t = int(ev["t"][0])
                last_t = int(ev["t"][-1])
            if tr is not None and len(tr):
                n_tr_pos += int(np.count_nonzero(tr["p"] == 1))
                n_tr_neg += int(np.count_nonzero(tr["p"] == 0))

    assert n > 0, f"{ef.path.name}: decoded no events"

    def check(name: str, actual: int, expected: int, tol: int = 0) -> None:
        assert abs(actual - expected) <= tol, (
            f"{ef.path.name}: {name} actual {actual} != expected {expected}"
            + (f" (tolerance {tol})" if tol else "")
        )

    # --- resolution ---
    width, height = meta["resolution"]
    if shape != [None, None]:
        assert shape == meta["resolution"], (
            f"{ef.path.name}: shape {shape} != metadata {meta['resolution']}"
        )
    assert x_max < width, f"{ef.path.name}: x max {x_max} >= width {width}"
    assert y_max < height, f"{ef.path.name}: y max {y_max} >= height {height}"

    # --- event counts (exact, save for the known Metavision-filtered files) ---
    rtol = 1e-3 if ef.path.name in _COUNT_TOLERANT_FILES else 0.0
    check("total events", n, meta["count"], int(meta["count"] * rtol))
    check("positive events", n_pos, meta["pos_count"], int(meta["pos_count"] * rtol))
    check("negative events", n_neg, meta["neg_count"], int(meta["neg_count"] * rtol))

    # --- external triggers ---
    tr_meta = meta.get("external_triggers", {"total": 0, "positive": 0, "negative": 0})
    if supports_triggers:
        check("total triggers", n_tr_pos + n_tr_neg, tr_meta["total"])
        check("positive triggers", n_tr_pos, tr_meta["positive"])
        check("negative triggers", n_tr_neg, tr_meta["negative"])
    else:
        # A format we cannot read triggers from must not be hiding any.
        assert tr_meta["total"] == 0, (
            f"{ef.path.name}: metadata declares {tr_meta['total']} triggers but "
            f"'{ef.path.suffix}' decoding cannot read them"
        )

    # --- timestamps trend upward overall (streams are not strictly monotonic) ---
    assert first_t is not None and first_t <= last_t, (
        f"{ef.path.name}: timestamps not increasing overall ({first_t} -> {last_t})"
    )
