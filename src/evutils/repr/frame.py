
import numba 
import numpy as np


@numba.njit
def frame_gray(events: np.ndarray, width: int = 1280, height: int = 720, dtype=np.uint8):
    '''
    Generate a frame from the events

    Parameters
    ----------
    ev : np.ndarray
        Array of events in the :class:`~evutils.types.Events` format
    width : int
        Width of the frame
    height : int
        Height of the frame
    fill : bool, optional
        If True, the non-zero values are set to 255, by default False
    dtype : np.dtype, optional
        Data type of the output array, by default np.uint8
    
    Returns
    -------
    out : np.ndarray
        A numpy array with the frame (height, width, 3)


    '''

    buffer = np.full((height, width), 128, dtype=dtype)


    for e in events:
        x = e['x']
        y = e['y']
        p = e['p']

        if p == 1:
            p = 255
        else:
            p = 0

        buffer[y, x] = p 

    return buffer


@numba.njit
def frame_rgb(ev: np.ndarray, width: int = 1280, height: int = 720, bg_color=np.array((0, 0, 0)), pos_color=np.array((255, 0, 0)), neg_color=np.array((0, 0, 255))):
    '''
    Generate a frame from the events

    Parameters
    ----------
    ev : np.ndarray
        Array of events in the :class:`~evutils.types.Events` format
    width : int
        Width of the frame
    height : int
        Height of the frame
    fill : bool, optional
        If True, the non-zero values are set to 255, by default True

    Returns
    -------
    out : np.ndarray
        A numpy array with the frame (height, width, 3)


    '''

    buffer = np.zeros((height, width, 3), dtype=np.uint8)
    buffer[:, :] = bg_color

    for e in ev:
        x = e['x']
        y = e['y']
        p = e['p']

        if p == 1:
            buffer[y, x] = pos_color
        else:
            buffer[y, x] = neg_color

    return buffer


@numba.njit
def frame_diff(ev: np.ndarray, width: int = 1280, height: int = 720, dtype=np.int8):
    '''
    Generate a differential frame from the events

    Parameters
    ----------
    ev : np.ndarray
        Array of events in the :class:`~evutils.types.Events` format
    width : int
        Width of the frame
    height : int
        Height of the frame
    dtype : np.dtype, optional
        Data type of the output array, by default np.int8
    Returns
    -------
    out : np.ndarray
        A numpy array with the frame (height, width, 3)


    '''

    buffer = np.zeros((height, width), dtype=dtype)

    for e in ev:
        x = e['x']
        y = e['y']
        p = e['p']

        if p == 1:
            buffer[y, x] += 1
        else:
            buffer[y, x] -= 1

    return buffer

