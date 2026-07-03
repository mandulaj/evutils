"""Data augmentation transforms for events.

Random and deterministic transformations for augmenting event streams
during training, such as spatial flips, time reversal, event dropout and
added noise.
"""


from ._drop import drop_random_events

__all__ = ["drop_random_events"]