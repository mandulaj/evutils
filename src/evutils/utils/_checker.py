"""Module providing utilities to check the validity of event arrays."""

import numpy as np

from ..types import Event_dtype


class EventsChecker():
    """Class to check if events are valid.
    
    Parameters
    ----------
    events : np.ndarray
        Array of events to check.

    """
    
    def __init__(self, events: 'np.ndarray | Any'):
        from ..types import SoaArray
        if isinstance(events, SoaArray):
            if events._aos_dtype != Event_dtype:
                raise ValueError("events must be of type Events")
        else:
            if events.dtype != Event_dtype:
                raise ValueError("events must be of type Events")

        self.events = events

    def is_sorted(self) -> bool:
        """Check if events are sorted by timestamp.
        
        Returns
        -------
        bool
            True if events are sorted in non-decreasing order of timestamps.

        """
        return bool(np.all(self.events['t'][1:] >= self.events['t'][:-1]))

    def has_valid_polarity(self) -> bool:
        """Check if events have valid polarity (0 or 1).
        
        Returns
        -------
        bool
            True if all events have polarity 0 or 1.

        """
        return bool(np.all((self.events['p'] == 0) | (self.events['p'] == 1)))
    
    def has_valid_x(self, width: int) -> bool:
        """Check if events have valid x coordinates.
        
        Parameters
        ----------
        width : int
            Maximum valid width (exclusive).
            
        Returns
        -------
        bool
            True if all events have x coordinates in [0, width).

        """
        return bool(np.all((self.events['x'] >= 0) & (self.events['x'] < width)))
    
    def has_valid_y(self, height: int) -> bool:
        """Check if events have valid y coordinates.
        
        Parameters
        ----------
        height : int
            Maximum valid height (exclusive).
            
        Returns
        -------
        bool
            True if all events have y coordinates in [0, height).

        """
        return bool(np.all((self.events['y'] >= 0) & (self.events['y'] < height)))
    
    def is_valid(self, width: int = 1280, height: int = 720) -> bool:
        """Check if all events are valid (sorted, valid polarity, and valid coordinates).
        
        Parameters
        ----------
        width : int, optional
            Maximum valid width, by default 1280.
        height : int, optional
            Maximum valid height, by default 720.
            
        Returns
        -------
        bool
            True if all checks pass.

        """
        return self.is_sorted() and self.has_valid_polarity() and self.has_valid_x(width) and self.has_valid_y(height)
    
    def __repr__(self) -> str:
        """Return string representation of the EventsChecker.
        
        Returns
        -------
        str
            String representation indicating the EventsChecker and number of events.

        """
        return f"EventsChecker(events={len(self.events)})"