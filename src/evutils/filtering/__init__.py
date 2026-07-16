"""Event stream filtering and cleanup.

Operations that select, mask or normalize events: spatial masking and
timestamp normalization. More filters (denoising, cropping, downsampling)
will land here.
"""

from ._masking import mask_events
from ._timestamp import normalize_ts


__all__ = [
    'mask_events',
    'normalize_ts'
]