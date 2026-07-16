"""Deprecated alias.

``EventsChecker`` moved to :mod:`evutils.types` (it validates the event types
defined there) and the ``utils`` grab-bag was removed. This shim re-exports
``EventsChecker`` and will be removed in a future release.
"""
import warnings

warnings.warn(
    "evutils.utils is deprecated and will be removed in a future release; "
    "import EventsChecker from evutils.types instead.",
    DeprecationWarning,
    stacklevel=2,
)

from evutils.types import EventsChecker  # noqa: F401,E402

__all__ = ["EventsChecker"]
