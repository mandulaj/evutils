
import numba 
import numpy as np


@numba.njit
# @numba.njit
def tore(events: np.ndarray, width: int = 1280, height: int = 720, n_events: int = 4, tau: int = 10_000, dtype=np.uint8):
    '''
    Generate a TORE from the events

    Parameters
    ----------
    ev : np.ndarray
        Array of events in the :class:`~evutils.types.Events` format
    width : int
        Width of the frame
    height : int
        Height of the frame
    n_events : int, optional
        Number of events to keep in the TORE, by default 4
    dtype : np.dtype, optional
        Data type of the output array, by default np.uint8
    
    
    Returns
    -------
    out : np.ndarray
        A numpy array with the frame (height, width, 3)

    [1] Baldwin, R. W., Liu, R., Almatrafi, M., Asari, V., & Hirakawa, K. (2022). Time-ordered recent event (tore) volumes for event cameras. IEEE Transactions on Pattern Analysis and Machine Intelligence, 45(2), 2519-2532.
    '''

    tore_fifo = np.zeros(((height, width, n_events, 2)), dtype=np.float32)

    if len(events) == 0:
        return tore_fifo


    tore_fifo_idx = np.full((height, width, 2), n_events - 1, dtype=np.int32)

    t_res = events[-1]['t']

    for i in range(len(events) -1, -1, -1):
        e = events[i]
        x = e['x']
        y = e['y']
        p = e['p']

        k_idx = tore_fifo_idx[y, x, p] 
        tore_fifo_idx[y, x, p] -= 1


        if k_idx >= 0:
            tore_fifo[y, x, k_idx, p] = np.exp(-(t_res - e['t'])/tau)

    
    return tore_fifo