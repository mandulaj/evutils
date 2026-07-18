"""Decomposed :class:`EventReader` (V2).

A drop-in reimplementation of :class:`evutils.io.EventReader` (V1) built as a
thin facade over a shared :class:`~evutils.io.v2.context.ReadContext`, per-mode
:mod:`~evutils.io.v2.strategies`, a :class:`~evutils.io.v2.cursor.SeekCursor`,
and a :class:`~evutils.io.v2.pacing.Pacer`. Same public surface and semantics
as V1; the monolith is decomposed for maintainability without perf regression.
"""

from .reader import EventReader

__all__ = ["EventReader"]
