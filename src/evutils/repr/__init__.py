"""Deprecated alias for :mod:`evutils.dense`.

The representations module was renamed ``repr`` -> ``dense`` (it holds dense,
fixed-size per-pixel encodings; a sibling ``sparse`` module may follow). This
shim re-exports everything from :mod:`evutils.dense` and will be removed in a
future release.
"""
import warnings

warnings.warn(
    "evutils.repr is deprecated and will be removed in a future release; "
    "use evutils.dense instead.",
    DeprecationWarning,
    stacklevel=2,
)

from evutils.dense import *  # noqa: F401,F403,E402
from evutils.dense import __all__  # noqa: F401,E402
