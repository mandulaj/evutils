
import numpy as np

from ..types import Events


class EventsChecker():
    """ Class to check if events are valid. """
    
    def __init__(self, events: np.ndarray):
        if events.dtype != Events:
            raise ValueError("events must be of type Events")

        self.events = events

    def is_sorted(self):
        return np.all(self.events['t'][1:] >= self.events['t'][:-1])

    def has_valid_polarity(self):
        return np.all((self.events['p'] == 0) | (self.events['p'] == 1))
    
    def has_valid_x(self, width: int):
        return np.all((self.events['x'] >= 0) & (self.events['x'] < width))
    
    def has_valid_y(self, height: int):
        return np.all((self.events['y'] >= 0) & (self.events['y'] < height))
    
    def is_valid(self, width: int = 1280, height: int = 720):
        return self.is_sorted() and self.has_valid_polarity() and self.has_valid_x(width) and self.has_valid_y(height)
    
    def __repr__(self):
        return f"EventsChecker(events={self.is_valid()})"