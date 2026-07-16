"""Data augmentation transforms for events.

torchvision/tonic-style transformations for event streams: random and
deterministic operations for augmenting data during training (drops, spatial
flips, jitter, time skew, refractory filtering, ...).

Each transform pairs a Numba-JIT ``functional`` kernel (dispatched over both
plain structured arrays and :class:`~evutils.types.EventArray`) with a
composable :class:`Transform` class. :class:`Compose` unwraps/rewraps events
only once around each contiguous block of JIT transforms.
"""

from .compose import Compose
from .transforms import (
    Transform,
    DropEvent,
    DropRandomEvents,
    DropEventByTime,
    RandomFlipLR,
    SpatialJitter,
    TimeSkew,
    TimeNormalize,
    TimeJitter,
    RefractoryPeriod,
)
import evutils.transforms.functional as functional

# Keep backward compatibility for users who imported functionals directly.
drop_random_events = functional.drop_random_events
drop_event = functional.drop_event

__all__ = [
    "Compose",
    "Transform",
    "DropEvent",
    "DropRandomEvents",
    "DropEventByTime",
    "RandomFlipLR",
    "SpatialJitter",
    "TimeSkew",
    "TimeNormalize",
    "TimeJitter",
    "RefractoryPeriod",
    "functional",
    "drop_random_events",
    "drop_event",
]
