"""Event stream filtering.

Operations that select or drop events by spatial region or content: spatial
masking today, with denoising, ROI and downsampling to follow. (Timestamp
normalization is a time *transform*, not a filter -- see
:mod:`evutils.transforms`.)
"""

from ._masking import mask_events


__all__ = [
    'mask_events',
]