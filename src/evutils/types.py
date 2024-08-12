


import numpy as np

import ctypes

__all__ = ['Event_dtype', 'Trigger_dtype']


#: A structured numpy dtype for event data.
#:
#: Fields:
#:
#: - `t` (np.int64): Timestamp of the event (us).
#: - `x` (np.uint16): X-coordinate.
#: - `y` (np.uint16): Y-coordinate.
#: - `p` (np.uint8): Polarity (0: off, 1: on).
Event_dtype = np.dtype([('t', np.int64), ('x', np.uint16), ('y', np.uint16), ('p', np.uint8)])


#: A structured numpy dtype for trigger data.
#:
#: Fields:
#:
#: - `t` (np.int64): Timestamp of the event (us).
#: - `p` (np.uint8):  Polarity (0: off, 1: on).
#: - `id` (np.uint8): Identifier.
Trigger_dtype = np.dtype([('t', np.int64), ('p', np.uint8), ('id', np.uint8)])


class Event(ctypes.Structure):
    _fields_ = [("t", ctypes.c_int64),
                ("x", ctypes.c_uint16),
                ("y", ctypes.c_uint16),
                ("p", ctypes.c_uint8)]



def is_monotonically_increasing(events: np.ndarray) -> bool:
    '''Checks if the event ts is monotonically increasing'''
    return bool(np.all(np.diff(events['t']) >= 0))



class Events(np.ndarray):
    '''
    Events
    '''
    def __new__(cls, input_array):
        obj = np.asarray(input_array, dtype=Event_dtype).view(cls)
        print("Creating Events")
        return obj
    
    def __array_finalize__(self, obj):
        if obj is None: 
            return
        print("Finalizing Events")
        

    def filter_by_time(self, start_time, end_time):                                                                                                                                                               
        """Returns events within the specified time window"""                                                                                                                                                     
        mask = (self['t'] >= start_time) & (self['t'] <= end_time)                                                                                                                                                
        return self[mask]  

    def is_monotonically_increasing(self):
        return is_monotonically_increasing(self)



class IndexedEvents(Events):
    def __new__(cls, input_array):
         # Use the __new__ method from Events                                                                                                                                                                      
         obj = super().__new__(cls, input_array)                                                                                                                                                                   
         return obj
    
    def __init__(self, input_array):                                                                                                                                                                              
         super().__init__(input_array)                                                                                                                                                                             
         # Build internal index                                                                                                                                                                                    
         self._build_internal_index()        
    

    def _build_internal_index(self):
        self.index = np.argsort(self['t'])

