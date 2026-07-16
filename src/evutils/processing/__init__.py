"""Deprecated alias.

The old ``processing`` module was split: ``mask_events`` moved to
:mod:`evutils.filtering` and ``normalize_ts`` moved to
:mod:`evutils.transforms` (it is a time transform, not a filter). This shim
re-exports both under their old names and will be removed in a future release.
"""
import warnings

warnings.warn(
    "evutils.processing is deprecated and will be removed in a future release; "
    "use evutils.filtering (mask_events) and evutils.transforms (normalize_ts) "
    "instead.",
    DeprecationWarning,
    stacklevel=2,
)

from evutils.filtering import mask_events  # noqa: F401,E402
from evutils.transforms.functional import normalize_ts  # noqa: F401,E402

__all__ = ["mask_events", "normalize_ts"]
