"""Data augmentation transforms for events.

Random and deterministic transformations for augmenting event streams
during training, such as spatial flips, time reversal, event dropout and
added noise.
"""


from .compose import Compose
from .transforms import Transform, DropRandomEvents
import evutils.transforms.functional as functional

# Keep backward compatibility for users who imported drop_random_events directly
drop_random_events = functional.drop_random_events

__all__ = [
    "Compose",
    "Transform",
    "DropRandomEvents",
    "functional",
    "drop_random_events"
]