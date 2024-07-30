
from ._writer import EventWriter

import numpy as np

 

from ._bin import EventWriter_Bin
from ._csv import EventWriter_Csv
from ._dat import EventWriter_Dat
from ._hdf5 import EventWriter_HDF5
from ._npz import EventWriter_Npz
from ._raw import EventWriter_RAW
from ._txt import EventWriter_Txt





# class EventWriter_Any(EventWriter):

#     def __init__(self, file, width=1280, height=720):
#         super().__init__(file, width, height)



