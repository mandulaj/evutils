#!/usr/bin/env python3
"""Bump the project version in one place, then commit and tag it.

pyproject.toml's ``[project].version`` is the single source of truth -- the C
extension (via CMake/SKBUILD_PROJECT_VERSION) and the docs both derive from it,
and the release workflow's check_version job refuses to publish if the git tag
disagrees with it. This helper keeps the tag and the file in lock-step.

Usage:
    python scripts/bump_version.py 0.3.16     # -> version 0.3.16, tag v0.3.16

It edits pyproject.toml, commits "chore: release vX.Y.Z", and creates tag
vX.Y.Z. It does NOT push and does NOT create the GitHub Release -- those stay
manual on purpose (the GitHub Release is the deliberate double-check gate).
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
_PYPROJECT = _ROOT / "pyproject.toml"
# PEP 440-ish: X.Y.Z with an optional pre/post/dev suffix.
_VERSION_RE = re.compile(r"^\d+\.\d+\.\d+([abc]|rc|\.post|\.dev)?\d*$")


def _run(*args: str) -> None:
    subprocess.run(args, cwd=_ROOT, check=True)


def main() -> int:
    if len(sys.argv) != 2:
        print(__doc__)
        return 2

    new = sys.argv[1].lstrip("v")
    if not _VERSION_RE.match(new):
        print(f"error: '{new}' is not a valid version (expected e.g. 0.3.16)")
        return 2

    # Refuse on a dirty tree so the release commit only contains the bump.
    dirty = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=_ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    if dirty:
        print("error: working tree is not clean; commit or stash first:\n" + dirty)
        return 1

    text = _PYPROJECT.read_text()
    new_text, n = re.subn(
        r'(?m)^(version\s*=\s*")[^"]*(")', rf"\g<1>{new}\g<2>", text, count=1
    )
    if n != 1:
        print('error: could not find a single version = "..." line in pyproject.toml')
        return 1
    _PYPROJECT.write_text(new_text)

    tag = f"v{new}"
    _run("git", "add", "pyproject.toml")
    _run("git", "commit", "-m", f"chore: release {tag}")
    # _run("git", "tag", tag)

    print(
        f"\nBumped to {new} and tagged {tag}.\n"
        f"Next:\n"
        f"  git push --follow-tags\n"
        f"  then create a GitHub Release from {tag} to trigger the build."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
