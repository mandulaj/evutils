import math
from typing import Union

import numpy as np

from evutils.types import SoaArray

from .functional._common import sample_range


class Transform:
    """Base class for all evutils transforms.

    Transforms should implement the `_forward_jit` method to allow zero-overhead
    composition inside a `Compose` pipeline.
    """
    #: Sensor size resolved from the input container for the current call, used
    #: as a fallback when a transform was constructed without an explicit one.
    #: See :meth:`bind_context`.
    _ctx_sensor_size = None

    def bind_context(self, events):
        """Capture per-call context (currently ``sensor_size``) from ``events``.

        Called by :meth:`__call__` and by :class:`Compose` before ``_forward_jit``
        so a transform can fall back to the container's ``sensor_size`` metadata
        when it was not given one explicitly. Stored transiently, not persisted.
        """
        self._ctx_sensor_size = getattr(events, "sensor_size", None)

    def _resolve_sensor_size(self):
        """Return the explicit ``sensor_size`` or the one bound from the events.

        Raises if neither is available.
        """
        ss = self.sensor_size if self.sensor_size is not None else self._ctx_sensor_size
        if ss is None:
            raise ValueError(
                f"{type(self).__name__} needs a sensor_size: pass it to the "
                f"constructor, or attach it to the events "
                f"(events.sensor_size = (W, H))."
            )
        return ss

    def __call__(self, events: Union[np.ndarray, SoaArray], target=None):
        """Applies the transform in a standalone manner."""
        from evutils.transforms.compose import unwrap_events, repack_events
        if len(events) == 0:
            if target is not None:
                return events, target
            return events

        self.bind_context(events)
        t, x, y, p = unwrap_events(events)
        t, x, y, p = self._forward_jit(t, x, y, p)
        events = repack_events(events, t, x, y, p)

        if target is not None:
            target = self._transform_target(target)
            return events, target
        return events

    def _forward_jit(self, t: np.ndarray, x: np.ndarray, y: np.ndarray, p: np.ndarray):
        """The pure array-math forward pass, ideally JIT-compiled."""
        raise NotImplementedError("Transforms must implement _forward_jit")

    def _transform_target(self, target):
        """Pure Python transformation of the target (e.g. bounding boxes).
        Defaults to doing nothing.
        """
        return target


class DropEvent(Transform):
    """Randomly drops each event with probability ``p``.

    Tonic-compatible replacement for ``DropRandomEvents`` (kept as an alias).
    Uses an independent Bernoulli mask per event, so the number dropped is
    binomially distributed around ``p * n`` rather than exactly ``p * n``.

    Parameters
    ----------
    p : float or tuple of float, optional
        Drop probability in ``[0, 1)``. A ``(lo, hi)`` tuple is sampled uniformly
        per call. Defaults to ``0.1``.
    """
    def __init__(self, p: Union[float, tuple] = 0.1):
        if not isinstance(p, (tuple, list)):
            if math.isnan(p) or p < 0 or p >= 1:
                raise ValueError("p must be in [0, 1)")
        self.p = p

    def _forward_jit(self, t, x, y, pol):
        from evutils.transforms.functional import _drop_random_events_jit
        prob = sample_range(self.p)
        if prob <= 0:
            return t, x, y, pol
        return _drop_random_events_jit(t, x, y, pol, prob)

    def __repr__(self):
        return f"{self.__class__.__name__}(p={self.p})"


# Backwards-compatible alias for the pre-rename class name.
DropRandomEvents = DropEvent


class DropEventByTime(Transform):
    """Drops every event inside one randomly-placed time window.

    Parameters
    ----------
    duration_ratio : float or tuple of float, optional
        Window length as a fraction of the recording span, in ``[0, 1)``. A
        ``(lo, hi)`` tuple is sampled uniformly per call. Defaults to ``0.2``.
    """
    def __init__(self, duration_ratio: Union[float, tuple] = 0.2):
        self.duration_ratio = duration_ratio

    def _forward_jit(self, t, x, y, p):
        from evutils.transforms.functional import _drop_by_time_jit
        ratio = sample_range(self.duration_ratio)
        if ratio <= 0:
            return t, x, y, p
        return _drop_by_time_jit(t, x, y, p, ratio)

    def __repr__(self):
        return f"{self.__class__.__name__}(duration_ratio={self.duration_ratio})"


class RandomFlipLR(Transform):
    """Flips events horizontally (``x' = width - 1 - x``) with probability ``p``.

    Parameters
    ----------
    sensor_size : tuple, optional
        ``(W, H)`` (or ``(W, H, P)``) sensor size; only the width ``W`` is used.
        If omitted, it is taken from the events' ``sensor_size`` metadata.
    p : float, optional
        Probability of performing the flip. Defaults to ``0.5``.
    """
    def __init__(self, sensor_size: tuple = None, p: float = 0.5):
        if not 0 <= p <= 1:
            raise ValueError("p must be in [0, 1]")
        self.sensor_size = sensor_size
        self.p = p

    def _forward_jit(self, t, x, y, pol):
        from evutils.transforms.functional import _flip_lr_jit
        if np.random.rand() <= self.p:
            width = int(self._resolve_sensor_size()[0])
            return _flip_lr_jit(t, x, y, pol, width)
        return t, x, y, pol

    def __repr__(self):
        return f"{self.__class__.__name__}(sensor_size={self.sensor_size}, p={self.p})"


class SpatialJitter(Transform):
    """Adds correlated Gaussian noise to event coordinates.

    Parameters
    ----------
    sensor_size : tuple, optional
        ``(W, H)`` (or ``(W, H, P)``) sensor size, used for clipping. Only needed
        when ``clip_outliers`` is True; if omitted, taken from the events'
        ``sensor_size`` metadata.
    var_x, var_y : float, optional
        Variances of the jitter in x and y. Default ``1.0``.
    sigma_xy : float, optional
        Off-diagonal covariance. Default ``0.0``.
    clip_outliers : bool, optional
        Drop events jittered outside the sensor. Default ``False``.
    """
    def __init__(self, sensor_size: tuple = None, var_x: float = 1.0, var_y: float = 1.0,
                 sigma_xy: float = 0.0, clip_outliers: bool = False):
        self.sensor_size = sensor_size
        self.var_x = var_x
        self.var_y = var_y
        self.sigma_xy = sigma_xy
        self.clip_outliers = clip_outliers

    def _forward_jit(self, t, x, y, p):
        from evutils.transforms.functional import _spatial_jitter_jit
        # Sensor size is only consulted when clipping; avoid requiring it otherwise.
        if self.clip_outliers:
            ss = self._resolve_sensor_size()
            width, height = int(ss[0]), int(ss[1])
        else:
            width, height = 0, 0
        return _spatial_jitter_jit(t, x, y, p, width, height, float(self.var_x),
                                   float(self.var_y), float(self.sigma_xy),
                                   bool(self.clip_outliers))

    def __repr__(self):
        return (f"{self.__class__.__name__}(sensor_size={self.sensor_size}, "
                f"var_x={self.var_x}, var_y={self.var_y}, sigma_xy={self.sigma_xy}, "
                f"clip_outliers={self.clip_outliers})")


class TimeSkew(Transform):
    """Rescales timestamps by an affine map ``t' = t * coefficient + offset``.

    Parameters
    ----------
    coefficient : float or tuple of float
        Multiplier applied to every timestamp. A ``(lo, hi)`` tuple is sampled
        uniformly per call.
    offset : float or tuple of float, optional
        Added after multiplication. Default ``0``.
    """
    def __init__(self, coefficient: Union[float, tuple],
                 offset: Union[float, tuple] = 0):
        self.coefficient = coefficient
        self.offset = offset

    def _forward_jit(self, t, x, y, p):
        from evutils.transforms.functional import _time_skew_jit
        coef = sample_range(self.coefficient)
        off = sample_range(self.offset)
        return _time_skew_jit(t, x, y, p, coef, off)

    def __repr__(self):
        return f"{self.__class__.__name__}(coefficient={self.coefficient}, offset={self.offset})"


class TimeJitter(Transform):
    """Adds Gaussian noise to each timestamp.

    Parameters
    ----------
    std : float
        Standard deviation of the timestamp noise.
    clip_negative : bool, optional
        Drop events with negative jittered timestamps. Default ``True``.
    sort_timestamps : bool, optional
        Re-sort by timestamp after jittering. Default ``False``.
    """
    def __init__(self, std: float, clip_negative: bool = True,
                 sort_timestamps: bool = False):
        self.std = std
        self.clip_negative = clip_negative
        self.sort_timestamps = sort_timestamps

    def _forward_jit(self, t, x, y, p):
        from evutils.transforms.functional import _time_jitter_jit
        return _time_jitter_jit(t, x, y, p, float(self.std),
                                bool(self.clip_negative), bool(self.sort_timestamps))

    def __repr__(self):
        return (f"{self.__class__.__name__}(std={self.std}, "
                f"clip_negative={self.clip_negative}, sort_timestamps={self.sort_timestamps})")


class RefractoryPeriod(Transform):
    """Enforces a per-pixel refractory period.

    Parameters
    ----------
    delta : int or tuple of int
        Refractory period in timestamp units. A ``(lo, hi)`` tuple is sampled
        uniformly per call. Events must be sorted by timestamp.
    """
    def __init__(self, delta: Union[int, tuple]):
        self.delta = delta

    def _forward_jit(self, t, x, y, p):
        from evutils.transforms.functional import _refractory_period_jit
        delta = int(sample_range(self.delta))
        return _refractory_period_jit(t, x, y, p, delta)

    def __repr__(self):
        return f"{self.__class__.__name__}(delta={self.delta})"
