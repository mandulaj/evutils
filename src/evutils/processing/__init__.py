"""Deprecated alias for :mod:`evutils.filtering`.

The module was renamed ``processing`` -> ``filtering``. This shim re-exports
everything from :mod:`evutils.filtering` and will be removed in a future
release.
"""
import warnings

warnings.warn(
    "evutils.processing is deprecated and will be removed in a future release; "
    "use evutils.filtering instead.",
    DeprecationWarning,
    stacklevel=2,
)

from evutils.filtering import *  # noqa: F401,F403,E402
from evutils.filtering import __all__  # noqa: F401,E402
