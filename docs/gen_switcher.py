#!/usr/bin/env python3
"""Generate ``switcher.json`` and mirror the stable docs to the site root.

The site published to the ``gh-pages`` branch holds one subdirectory per docs
version -- ``vX.Y.Z`` for each release plus an optional ``dev`` built from the
``dev`` branch. This script scans that site directory and:

* writes ``switcher.json`` -- the version list consumed by pydata-sphinx-theme's
  built-in version switcher. The newest final release is marked ``preferred``
  (the "stable" one); ``dev`` is listed first but never preferred.
* mirrors the stable version's built site into the root, so
  ``owner.github.io/repo/`` serves the latest stable docs directly (clean URLs)
  while ``owner.github.io/repo/vX.Y.Z/`` serves a specific version. GitHub Pages
  does not follow symlinks, so this is a copy. Version subdirs and switcher.json
  are preserved; other root files are refreshed from the stable build. A side
  effect is that the root gains its own ``_static/``, so README's absolute
  ``/_static/logo`` link (used so the logo renders on GitHub/PyPI too) resolves
  for every version.

Usage:
    gen_switcher.py <base_url> <site_dir>

``base_url`` is the Pages root, e.g. ``https://owner.github.io/repo``.
``site_dir`` is scanned for version subdirs AND is where output is written.
"""
from __future__ import annotations

import json
import re
import shutil
import sys
from pathlib import Path

# vMAJOR.MINOR.PATCH with an optional suffix (e.g. v1.2.3rc1, v1.2.3.post1).
_VER_RE = re.compile(r"^v(\d+)\.(\d+)\.(\d+)(.*)$")

# Root entries that are NOT part of a mirrored build and must survive the mirror.
# `coverage/` is published independently by the Coverage workflow
# (docs/deploy_coverage.sh) and must not be wiped when the docs mirror refreshes
# the root -- otherwise the README's coverage badge/link 404s after any deploy.
_ROOT_KEEP = {"switcher.json", ".nojekyll", ".git", "coverage"}


def _release_key(name: str) -> tuple:
    """Sort key for release dir names. Final releases outrank pre-releases of
    the same x.y.z (e.g. v1.2.3 > v1.2.3rc1)."""
    m = _VER_RE.match(name)
    major, minor, patch, suffix = int(m[1]), int(m[2]), int(m[3]), m[4]
    return (major, minor, patch, suffix == "", suffix)


def _mirror_to_root(site: Path, version: str, version_dirs: set[str]) -> None:
    """Copy the built site under ``site/<version>`` into ``site`` (root),
    replacing prior root files but never touching version dirs or _ROOT_KEEP."""
    protected = version_dirs | _ROOT_KEEP
    for p in site.iterdir():
        if p.name in protected:
            continue
        if p.is_dir():
            shutil.rmtree(p)
        else:
            p.unlink()

    src = site / version
    for p in src.iterdir():
        dst = site / p.name
        if p.is_dir():
            shutil.copytree(p, dst)
        else:
            shutil.copy2(p, dst)


def main() -> int:
    base_url = sys.argv[1].rstrip("/")
    site = Path(sys.argv[2])

    version_dirs = {p.name for p in site.iterdir()
                    if p.is_dir() and (_VER_RE.match(p.name) or p.name == "dev")}
    releases = sorted((n for n in version_dirs if _VER_RE.match(n)),
                      key=_release_key, reverse=True)
    has_dev = "dev" in version_dirs

    # "stable" is the newest FINAL release (no rc/dev/post suffix); a
    # pre-release is never the preferred version.
    finals = [r for r in releases if _VER_RE.match(r).group(4) == ""]
    preferred = finals[0] if finals else None

    entries: list[dict] = []
    if has_dev:
        dev = {"name": "dev (main)", "version": "dev", "url": f"{base_url}/dev/"}
        if preferred is None:  # nothing stable yet -> dev is the default
            dev["preferred"] = True
        entries.append(dev)
    for v in releases:
        entry = {"name": v, "version": v, "url": f"{base_url}/{v}/"}
        if v == preferred:
            entry["name"] = f"{v} (stable)"
            entry["preferred"] = True
        entries.append(entry)

    (site / "switcher.json").write_text(json.dumps(entries, indent=2) + "\n")

    # Mirror the best-available version to the root: stable, else newest
    # pre-release, else dev.
    root_version = preferred or (releases[0] if releases else
                                 ("dev" if has_dev else None))
    if root_version:
        _mirror_to_root(site, root_version, version_dirs)

    print(f"switcher.json: {len(entries)} entries; root mirrors {root_version}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
