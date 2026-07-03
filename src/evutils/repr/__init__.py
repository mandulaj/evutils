"""Dense representations of event data.

Convert sparse event streams into dense tensors — voxel grids, time
surfaces, event frames and other fixed-size representations for downstream
models.
"""


from ._histogram import histogram, wedge_histogram
from ._voxel import voxel_histogram
from ._timesurface import timesurface
from ._frame import frame_diff, frame_rgb, frame_gray
from ._tore import tore


__all__ = [
    'histogram',
    'wedge_histogram',
    'voxel_histogram',
    'timesurface',
    'frame_diff',
    'frame_rgb',
    'frame_gray',
    'tore'
]
