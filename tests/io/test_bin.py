"""The ``.bin`` format is a reserved stub: reading and writing must fail with
an explicit :class:`NotImplementedError` (never a silent no-op or an obscure
downstream crash)."""
import numpy as np
import pytest

from evutils.io import EventReader, EventWriter
from evutils.types import Event_dtype


def test_bin_read_raises_not_implemented(tmp_path):
    p = tmp_path / "events.bin"
    p.write_bytes(b"\x00" * 64)
    with pytest.raises(NotImplementedError, match="not implemented"):
        EventReader(str(p))


def test_bin_write_raises_not_implemented(tmp_path):
    ev = np.zeros(10, dtype=Event_dtype)
    with pytest.raises(NotImplementedError, match="not implemented"):
        with EventWriter(str(tmp_path / "events.bin")) as w:
            w.write(ev)
