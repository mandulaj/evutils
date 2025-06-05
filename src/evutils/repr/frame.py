
import numba 
import numpy as np


@numba.njit
def frame(events: np.ndarray, width: int = 1280, height: int = 720, dtype=np.uint8):
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