"""Visualization tools for event camera data.

This package contains various visualization tools for event streams,
including 3D plots, Open3D point clouds, and video reconstruction.

Submodules are imported lazily (via ``__getattr__``, mirroring
:mod:`evutils`) so ``import evutils.vis`` does not eagerly pull in the heavy
``cv2`` / ``matplotlib`` / ``open3d`` / ``torch`` stacks. Access
``evutils.vis.plot3d`` / ``evutils.vis.reconstructor`` / ``evutils.vis.open3d``
to resolve them on first use.
"""
import importlib

__all__ = ['reconstructor', 'plot3d', 'open3d']


def __getattr__(name: str):
    if name in __all__:
        return importlib.import_module(f".{name}", __name__)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return __all__
