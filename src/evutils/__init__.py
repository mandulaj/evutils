"""evutils: utilities for event-based data.

Event-based data processing and visualization utilities.

To get started::

    import evutils
    ...



Notes
-----
Anything worth calling out up front.



"""

import importlib.metadata

try:
    __version__ = importlib.metadata.version("evutils")
except importlib.metadata.PackageNotFoundError:
    # Fallback if the package is run without being installed
    __version__ = "dev"

from .types import EventArray, TriggerArray, Event_dtype, Trigger_dtype

__all__ = [
    'chunking', 
    'dataset', 
    'io', 
    'jit',
    'processing', 
    'random', 
    'repr',
    'torch',
    'transforms',
    'types', 
    'utils', 
    'vis', 
    '__version__'
]

