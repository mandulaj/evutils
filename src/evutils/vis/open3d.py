"""Open3D visualization utilities for event camera data.

This module provides functions to visualize event streams in 3D (x, y, time)
using Open3D.
"""
import open3d as o3d
import numpy as np
from matplotlib import pyplot as plt 

def o3d_draw_events(events: np.ndarray) -> None:
    """Visualizes events using Open3D.

    Parameters
    ----------
    events : np.ndarray
        Array of events with fields 'x', 'y', 't', and 'p'.
        'x' and 'y' are the pixel coordinates, 't' is the timestamp, and 'p' is the polarity.

    Returns
    -------
    None

    """
    if len(events) == 0:
        return

    # Draw X, Y, T as a pointcloud 
    pdc = o3d.geometry.PointCloud()

    time_diff = np.max(events['t']) - np.min(events['t'])
    # Normalize time to range 0..3000
    norm_time = (events['t'] - np.min(events['t'])) / time_diff * 3000

    pdc.points = o3d.utility.Vector3dVector(np.column_stack((events['x'], events['y'], norm_time)))

    # Create a color map based on the 'p'
    # p can be either 0 or 1, we use it to color the points witht eh Spectral colormap
    colors = plt.cm.Spectral(events['p'].astype(np.float32))[:, :3]  # Use only RGB channels

    pdc.colors = o3d.utility.Vector3dVector(colors)

    o3d.visualization.draw_geometries([pdc])