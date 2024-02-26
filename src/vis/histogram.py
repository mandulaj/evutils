import numba 
import numpy as np


@numba.njit
def gen_frame(ev: np.ndarray, width: int, height: int, fill=True):

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
