import numba 
import numpy as np


@numba.njit
def histogram(events: np.ndarray, width: int = 1280, height: int = 720, fill=False, dtype=np.uint8):
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

    buffer = np.zeros((height, width, 3), dtype=dtype)

    clip = 255

    # clip = 255 if dtype == np.uint8 else 65535

    for e in events:
        x = e['x']
        y = e['y']
        p = e['p']

        if p == 1:
            p = 0 # Red
        else:
            p = 2 # Blue
        


        if buffer[y, x, p] < clip:
            buffer[y, x, p] += 1

    # If the fill flag is set, we fill the non-zero values with 255
    if fill:
        for x in range(0, width):
            for y in range(0, height):
                for c in range(0, 3):
                    if buffer[y, x, c] > 0:
                        buffer[y, x, c] = 255

    return buffer


@numba.njit
def wedge_histogram(events: np.ndarray, width: int = 1280, height: int = 720, tl: float = 30e6, dtype=np.uint8):   
    '''
    Generate a wedge frame frame from the events

    Parameters
    ----------
    ev : np.ndarray
        Array of events in the :class:`~evutils.types.Events` format
    buffer : np.ndarray
        A numpy array with the frame (height, width, 3)
    tl : int, optional
        Duration of the frame in us, by default 30e6
    
    Returns
    -------
    out : np.ndarray
        A numpy array with the wedge frame (height, width, 3)


    '''
    buffer = np.zeros((height, width, 3), dtype=dtype)

    ts_norm = (events['t'] - events['t'][0])/tl

    # print(len(ev))

    sel = (height-events['y'])/height > ts_norm - 0.1
    ev_sel = events[sel]
    # print(len(ev_sel))
    # print((720-ev['y'])/720, ts_norm, sel)


    for e in ev_sel:
        x = e['x']
        y = e['y']
        p = e['p']


        if p == 1:
            p = 0
        else:
            p = 2
        
        if buffer[y, x, p] < 128:
            buffer[y, x, p] += 255

    return buffer