#!/usr/bin/env python3
"""Generate ``switcher.json`` + a root ``index.html`` for the versioned docs.

The site published to the ``gh-pages`` branch holds one subdirectory per docs
version -- ``vX.Y.Z`` for each release plus an optional ``dev`` built from the
``dev`` branch. This script scans that site directory and writes:

* ``switcher.json`` -- the version list consumed by pydata-sphinx-theme's
  built-in version switcher. The newest release is marked ``preferred`` (the
  "stable" one); ``dev`` is listed first but never preferred.
* ``index.html`` -- a redirect at the site root that sends visitors to the
  newest release (or to ``dev`` if no release exists yet).

Usage:
    gen_switcher.py <base_url> <site_dir>

``base_url`` is the Pages root, e.g. ``https://owner.github.io/repo``.
``site_dir`` is scanned for version subdirs AND is where the two files are
written.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

# vMAJOR.MINOR.PATCH with an optional suffix (e.g. v1.2.3rc1, v1.2.3.post1).
_VER_RE = re.compile(r"^v(\d+)\.(\d+)\.(\d+)(.*)$")

_REDIRECT = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta http-equiv="refresh" content="0; url=./{target}">
  <link rel="canonical" href="./{target}">
  <title>Redirecting…</title>
</head>
<body>Redirecting to the <a href="./{target}">latest documentation</a>.</body>
</html>
"""


def _release_key(name: str) -> tuple:
    """Sort key for release dir names. Final releases outrank pre-releases of
    the same x.y.z (e.g. v1.2.3 > v1.2.3rc1)."""
    m = _VER_RE.match(name)
    major, minor, patch, suffix = int(m[1]), int(m[2]), int(m[3]), m[4]
    return (major, minor, patch, suffix == "", suffix)


def main() -> int:
    base_url = sys.argv[1].rstrip("/")
    site = Path(sys.argv[2])

    names = {p.name for p in site.iterdir() if p.is_dir()}
    releases = sorted((n for n in names if _VER_RE.match(n)),
                      key=_release_key, reverse=True)
    has_dev = "dev" in names

    # "stable" is the newest FINAL release (no rc/dev/post suffix); a
    # pre-release is never the preferred version or the redirect target.
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

    # Redirect the site root to stable; fall back to newest pre-release, then dev.
    target = preferred or (releases[0] if releases else ("dev" if has_dev else None))
    if target:
        (site / "index.html").write_text(_REDIRECT.format(target=f"{target}/"))

    print(f"switcher.json: {len(entries)} entries; root redirect -> {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
