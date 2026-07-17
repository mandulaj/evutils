"""Import-level smoke tests for the vis module.

The reconstructors need torch (+ model weights) and plot3d needs a display, so
functional coverage lives outside CI; these tests only pin that importing the
package does not break and skips cleanly when optional deps are missing.
"""
import pytest


def test_vis_imports():
    vis = pytest.importorskip("evutils.vis")
    assert hasattr(vis, "reconstructor")


def test_plot3d_importable():
    pytest.importorskip("cv2")
    pytest.importorskip("matplotlib")
    from evutils.vis import plot3d
    assert callable(plot3d.plot_3d)
