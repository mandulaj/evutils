import numba 
import numpy as np

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


from typing import Union, Optional
from matplotlib.colors import Colormap


def plot_3d(events: np.ndarray, 
            width: int =1280, 
            height: int = 720, 
            colormap: Union[str, Colormap] ='Spectral', 
            fig: Optional[plt.Figure] = None, 
            ax: Optional[Axes3D] = None) -> tuple:   
    """Plot a 3D scatter plot of events.
    
    Parameters
    ----------
    events : np.ndarray
        Array of events with fields 't', 'x', 'y', and 'p'.
    width : int, optional
        Width of the event frame, by default 1280.
    height : int, optional
        Height of the event frame, by default 720.
    colormap : Union[str, Colormap], optional
        Colormap to use for the event polarities, by default 'Spectral'.
    fig : Optional[plt.Figure], optional
        Matplotlib figure to plot on, by default None (a new figure will be created).
    ax : Optional[Axes3D], optional
        Matplotlib 3D axis to plot on, by default None (a new axis will be created).
    Returns
    -------
    tuple
        A tuple containing the figure and axis objects.
    Raises
    ------
    ValueError
        If the colormap is not found or invalid.
    
    """
    ts = events['t'] / 1000.0  # Convert timestamps to seconds

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