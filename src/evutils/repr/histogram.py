import numba 
import numpy as np


@numba.njit
def gen_frame(ev: np.ndarray, width: int, height: int, fill=True):
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

    for e in ev:
        x = e['x']
        y = e['y']
        p = e['p']

        if p == 1:
            p = 0 # Red
        else:
            p = 2 # Blue
        
        if buffer[y, x, p] < 255:
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
def wedge_frame(ev, buffer, tl = 30e6):
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
    buffer[:] = 0

    ts_norm = (ev['t'] - ev['t'][0])/tl

    # print(len(ev))

    sel = (720-ev['y'])/720 > ts_norm - 0.1
    ev_sel = ev[sel]
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