"""Matplotlib 3D visualization utilities for event data.

This module provides functions to plot event streams, 3D histograms,
and time surfaces using matplotlib's 3D plotting capabilities.
"""
import numba 
import numpy as np
import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


from typing import Union, Optional
from matplotlib.colors import Colormap
from matplotlib.figure import Figure


def plot_3d(events: np.ndarray, 
            width: int =1280, 
            height: int = 720, 
            colormap: Union[str, Colormap] ='Spectral', 
            fig: Optional[Figure] = None, 
            ax: Optional[Axes3D] = None) -> tuple[Optional[Figure], Optional[Axes3D]]:   
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

    if len(events) == 0:
        return fig, ax

    # Plot as points
    ax.scatter(ts, events['x'], events['y'], c=colors, s=1, alpha=0.2)

    # Labels
    ax.set_xlabel('time (ms)')
    ax.set_ylabel('x')
    ax.set_zlabel('y')
    ax.set_title('Event Stream 3D Plot')
    ax.set_xlim([ts.min(), ts.max()])
    ax.set_ylim([0, width])
    ax.set_zlim([height, 0])
    ax.set_box_aspect([max(width,height), width, height])  # Aspect ratio

    return fig, ax


def plot_3d_histogram(histogram: np.ndarray,
                      down_sample: int = 4,
                      fig: Optional[Figure] = None,
                      ax: Optional[Axes3D] = None) -> tuple[Optional[Figure], Optional[Axes3D]]:
    """Plot a 3D histogram of events.

    Parameters
    ----------
    histogram : np.ndarray
        3D histogram array with shape (depth, height, width).
    down_sample : int, optional
        Downsampling factor for the histogram, by default 4.
    fig : Optional[plt.Figure], optional
        Matplotlib figure to plot on, by default None (a new figure will be created).
    ax : Optional[Axes3D], optional
        Matplotlib 3D axis to plot on, by default None (a new axis will be created).

    Returns
    -------
    tuple
        A tuple containing the figure and axis objects.

    """
    # Create figure/axis if not provided
    if ax is None:
        fig = plt.figure(figsize=(10, 7)) if fig is None else fig
        ax = fig.add_subplot(111, projection='3d')
    # Create a meshgrid for the histogram

    assert histogram.ndim == 4, "Histogram must be a 4D array."
    n_bins, height, width, ch = histogram.shape

    height_downsampled = height // down_sample
    width_downsampled = width // down_sample
    x, y = np.meshgrid(np.linspace(0, width - 1, width_downsampled),
                       np.linspace(0, height - 1, height_downsampled))
    

    for bin in range(n_bins):
        print(f"Plotting bin {bin+1}/{n_bins}")
        image = histogram[bin]

        # Downsample the image for better visualization
        image = cv2.resize(image, (width_downsampled, height_downsampled), interpolation=cv2.INTER_LINEAR)

        # Conver to RGBA
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGBA)

        # Set the A channel to 0 where the pixel is black
        image[:, :, 3] = np.where(np.all(image[:, :, :3] == 0, axis=-1), 0, 255)



        z = np.full(x.shape, bin)

        # Plot the surface
        ax.plot_surface(z, x, y, facecolors=image/255, rstride=1, cstride=1, shade=False)

        # Create a meshgrid for the histogram

    ax.set_xlabel('bins')
    ax.set_ylabel('x')
    ax.set_zlabel('y')
    ax.set_title('Voxel Histogram')
    ax.set_xlim([0, n_bins])
    ax.set_ylim([0, width])
    ax.set_zlim([height, 0])
    ax.set_box_aspect([max(width,height), width, height])  # Aspect ratio

    return fig, ax




def plot_3d_timesurface(events: np.ndarray, 
            width: int =1280, 
            height: int = 720,
            tau: int = 10_000,
            fig: Optional[Figure] = None, 
            ax: Optional[Axes3D] = None) -> tuple[Optional[Figure], Optional[Axes3D]]:   
    """Plot a 3D time surface of events.

    Parameters
    ----------
    events : np.ndarray
        Array of events with fields 't', 'x', 'y', and 'p'.
    width : int, optional
        Width of the event frame, by default 1280.
    height : int, optional
        Height of the event frame, by default 720.
    tau : int, optional
        Time constant for the time surface exponential decay in microseconds, by default 10_000.
    fig : Optional[plt.Figure], optional
        Matplotlib figure to plot on, by default None (a new figure will be created).
    ax : Optional[Axes3D], optional
        Matplotlib 3D axis to plot on, by default None (a new axis will be created).

    Returns
    -------
    tuple
        A tuple containing the figure and axis objects.

    """
    ts = events['t'] # Convert timestamps to seconds

    # Create figure/axis if not provided
    if ax is None:
        fig = plt.figure(figsize=(10, 7)) if fig is None else fig
        ax = fig.add_subplot(111, projection='3d')

    if len(events) == 0:
        return fig, ax

    ts_ref = ts[-1]  # Reference timestamp for normalization
    
    p_sign = (events['p'].astype(np.float32) * 2 - 1)

    colors = np.exp(-(ts_ref - ts) / tau) * p_sign  # Normalize polarity

    ts = ts / 1_000  # Convert timestamps to milliseconds
    # Plot as points
    ax.scatter(ts, events['x'], events['y'], c=colors, s=1, alpha=0.2)

    # Labels
    ax.set_xlabel('time (ms)')
    ax.set_ylabel('x')
    ax.set_zlabel('y')
    ax.set_title('Event Stream 3D Plot')
    ax.set_xlim([ts.min(), ts.max()])
    ax.set_ylim([0, width])
    ax.set_zlim([height, 0])
    ax.set_box_aspect([max(width,height), width, height])  # Aspect ratio
    
    return fig, ax