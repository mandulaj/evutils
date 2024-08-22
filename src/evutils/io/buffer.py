import numpy as np
from ..types import Event_dtype

class EventRingBuffer():
    def __init__(self, size:int, dtype=Event_dtype):
        self.size = size
        self.buffer = np.empty(size, dtype=dtype)
        self.dtype = dtype
        self.start = 0
        self.end = 0
    
    def __len__(self) -> int:
        return self.end - self.start
    
    @property
    def capacity(self) -> int:
        return self.size - self.end
    
    def append(self, data: np.ndarray):
        if len(data) > self.capacity:
            self.rotate() # Try rotating to get more space
        if len(data) > self.capacity:
            print(f"Data: {len(data)}, Capacity: {self.capacity}, start: {self.start}, end: {self.end}")
            raise ValueError(f"Ring Buffer is full, can't append {len(data)} elememnts when {self.capacity} space left.")

        
        self.buffer[self.end:self.end+len(data)] = data
        self.end += len(data)
        

    def advance(self, items:int):
        if self.start + items > self.size:
            raise ValueError(f"Cant advance beyond the buffer size {self.end}+{items}<{self.size}")
        self.start += items

    def view(self) -> np.ndarray:
        return self.buffer[self.start:self.end]
    
    def reset(self):
        self.start = 0
        self.end = 0

    def rotate(self):
        cur_len = len(self)
        self.buffer[0:cur_len] = self.view()
        self.start = 0
        self.end = cur_len

    def __getitem__(self, key):
        if isinstance(key, slice):
            
            # Deal if negative indexes

            if key.start is None:
                start_idx = self.start
            else:
                start_idx = self.start + key.start if key.start >= 0 else self.end + key.start
            
            if key.stop is None:
                end_idx = self.end
            else:
                end_idx = self.start + key.stop if key.stop >= 0 else self.end + key.stop

            assert start_idx <= end_idx

            return self.buffer[start_idx: end_idx: key.step]
        elif isinstance(key, int):

            idx = self.start + key if key >= 0 else self.end + key
            
            return self.buffer[idx]
        else:
            raise ValueError(f"Unsupported key in Ringbuffer {type(key)}")

    def __repr__(self) -> str:
        return str(self.view())
