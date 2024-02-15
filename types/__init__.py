


import numpy as np


Events = np.dtype([('t', np.uint64), ('x', np.uint16), ('y', np.uint16), ('p', np.uint8)])
Triggers = np.dtype([('t', np.uint64), ('p', np.uint8), ('id', np.uint8)])