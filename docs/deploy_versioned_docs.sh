#!/usr/bin/env bash
# Publish one docs version into the versioned site on the gh-pages branch.
#
# Uses only first-party pieces: plain git + the auto-provided GITHUB_TOKEN (no
# third-party GitHub Actions). The gh-pages branch holds every version in its
# own subdir; this script replaces ONLY the given version's dir, regenerates
# switcher.json, mirrors the stable version to the site root, and pushes -- so
# all other versions survive.
#
# Usage: deploy_versioned_docs.sh <version> <pages_base_url> <built_html_dir>
#   version         e.g. v0.3.16  or  dev
#   pages_base_url  e.g. https://owner.github.io/repo
#   built_html_dir  e.g. docs/build/html
#
# Requires env: GITHUB_TOKEN, GITHUB_REPOSITORY (both set by GitHub Actions).
set -euo pipefail

VERSION="$1"
PAGES_BASE="$2"
HTML_DIR="$3"

: "${GITHUB_TOKEN:?GITHUB_TOKEN is required}"
: "${GITHUB_REPOSITORY:?GITHUB_REPOSITORY is required}"

REMOTE="https://x-access-token:${GITHUB_TOKEN}@github.com/${GITHUB_REPOSITORY}.git"

# Fetch the current site (shallow). On the very first deploy the branch does not
# exist yet, so start an empty repo on the gh-pages branch instead.
rm -rf site
if git clone --depth 1 --branch gh-pages "$REMOTE" site 2>/dev/null; then
  echo "Cloned existing gh-pages branch."
else
  echo "gh-pages branch not found; initializing a new one."
  mkdir site
  git -C site init -q -b gh-pages
  git -C site remote add origin "$REMOTE"
fi

# Replace just this version's directory.
rm -rf "site/${VERSION}"
mkdir -p "site/${VERSION}"
cp -a "${HTML_DIR}/." "site/${VERSION}/"

# Regenerate the switcher and mirror the stable version to the site root
# (owner.github.io/repo -> latest stable; /vX.Y.Z/ -> that version).
python docs/gen_switcher.py "$PAGES_BASE" site

# Sphinx output has _static/_sources dirs; disable Jekyll so they aren't hidden.
touch site/.nojekyll

git -C site add -A
if git -C site \
     -c user.name="github-actions[bot]" \
     -c user.email="jakub.aludnam+bot@gmail.com" \
     commit -q -m "docs: deploy ${VERSION}"; then
  git -C site push -q origin gh-pages
  echo "Deployed ${VERSION}."
else
  echo "No documentation changes for ${VERSION}; nothing to deploy."
fi
