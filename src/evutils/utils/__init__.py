"""General-purpose helpers.

Miscellaneous utilities that don't belong to a specific area, including
``EventsChecker`` for validating that an event array is well-formed —
sorted timestamps, valid polarities, and coordinates within the sensor size.
"""

from ._checker import EventsChecker