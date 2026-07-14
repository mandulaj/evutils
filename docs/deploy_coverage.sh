#!/usr/bin/env bash
# Publish the coverage badge + HTML reports into the gh-pages /coverage/ subdir.
#
# Uses only first-party pieces: plain git + the auto-provided GITHUB_TOKEN (no
# third-party GitHub Actions). Shares the gh-pages branch with the versioned
# docs, so run it under the same `gh-pages-deploy` concurrency group -- it only
# touches site/coverage/, leaving every docs version untouched.
#
# Usage: deploy_coverage.sh <coverage_dir>
#   coverage_dir  local dir whose contents become /coverage/ on the site
#                 (e.g. badge.svg + htmlcov/ + htmlcov-c/)
#
# Requires env: GITHUB_TOKEN, GITHUB_REPOSITORY (both set by GitHub Actions).
set -euo pipefail

SRC="$1"

: "${GITHUB_TOKEN:?GITHUB_TOKEN is required}"
: "${GITHUB_REPOSITORY:?GITHUB_REPOSITORY is required}"

REMOTE="https://x-access-token:${GITHUB_TOKEN}@github.com/${GITHUB_REPOSITORY}.git"

# Fetch the current site (shallow). On the very first deploy the branch may not
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

# Replace only the coverage subdir; the versioned docs dirs are left alone.
rm -rf site/coverage
mkdir -p site/coverage
cp -a "${SRC}/." site/coverage/

# Sphinx output relies on this too; harmless to (re)assert.
touch site/.nojekyll

git -C site add -A
if git -C site \
     -c user.name="github-actions[bot]" \
     -c user.email="41898282+github-actions[bot]@users.noreply.github.com" \
     commit -q -m "coverage: update reports + badge"; then
  git -C site push -q origin gh-pages
  echo "Coverage published to /coverage/."
else
  echo "No coverage changes; nothing to deploy."
fi
