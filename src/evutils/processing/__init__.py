"""Event stream processing and filtering.

Operations that transform or filter events — denoising, spatial/temporal
cropping, downsampling and similar steps.
"""

from ._masking import mask_events
from ._utils import normalize_ts


__all__ = [
    'mask_events',
    'normalize_ts'
]