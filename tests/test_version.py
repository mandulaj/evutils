"""Version consistency between the Python package and the compiled C backend.

The C version is single-sourced from pyproject at build time: scikit-build-core
sets ``SKBUILD_PROJECT_VERSION`` -> the ``EVUTILS_VERSION`` macro ->
``evutils_version()`` (see CMakeLists.txt). So a package built through the normal
wheel / editable path carries a native library stamped with the exact package
version; a mismatch means a stale or wrong native library got bundled -- exactly
what this guards against (e.g. an old build/ shared lib shadowing a new wheel).

A standalone ``cmake`` build (scripts/coverage.sh, manual dev builds) cannot set
``SKBUILD_PROJECT_VERSION`` and falls back to a dev sentinel; there is nothing to
compare against then, so the check skips. That sentinel never appears in a
scikit-build wheel, so skipping on it cannot mask a real packaging mismatch.
"""
import pytest

import evutils

# Non-scikit-build sentinels: CMakeLists.txt fallback, and the csrc #ifndef
# default used only when compiled entirely without CMake.
_DEV_SENTINELS = {"0.0.0+dev", "0.0.1"}


def _native_version() -> str:
    from evutils.io._native_core import lib
    return lib().evutils_version().decode()


def test_native_backend_version_matches_package() -> None:
    native = _native_version()
    assert native, "evutils_version() returned an empty string"
    if native in _DEV_SENTINELS:
        pytest.skip(
            f"native lib not version-stamped (standalone-build sentinel {native!r}); "
            "built outside scikit-build-core"
        )
    assert native == evutils.__version__, (
        f"C backend version {native!r} != package version {evutils.__version__!r} -- "
        "stale or mismatched native library bundled"
    )
