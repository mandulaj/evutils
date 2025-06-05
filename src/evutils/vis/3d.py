import numba 
import numpy as np

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


from typing import Union, Optional
from matplotlib.colors import Colormap


def plot_3d(events: np.ndarray, width: int =1280, height: int = 720, colormap: Union[str, Colormap] ='Spectral'):
    """Plot a 3D scatter plot of events."""
    ts = events['t']

    # Normalize timestamps
    ts = ts / 1000.0  # now in milliseconds

    # Create color map: red for polarity=0, blue for polarity=1
     # Resolve colormap
    if isinstance(colormap, str):
        colormap = plt.get_cmap(colormap)
    if colormap is None:
        raise ValueError(f"Colormap not found or invalid.")
    
    colors = colormap(events['p'].astype(np.float32))  # Normalize polarity to [0, 1] for colormap

    # Create figure/axis if not provided
    if ax is None:
        fig = plt.figure(figsize=(10, 7)) if fig is None else fig
        ax = fig.add_subplot(111, projection='3d')

    # Plot as points
    ax.scatter(ts, events['x'], events['y'], c=colors, s=1, alpha=0.2)

    # Labels
    ax.set_xlabel('time (ms)')
    ax.set_ylabel('x')
    ax.set_zlabel('y')
    ax.set_title('Event Stream 3D Plot')
    ax.set_xlim([ts.min(), ts.max()])
    ax.set_ylim([0, width])
    ax.set_zlim([0, height])
    ax.set_box_aspect([max(width,height), width, height])  # Aspect ratio

    return fig, ax