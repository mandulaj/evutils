#!/usr/bin/env bash
# Build the C sources with gcov instrumentation, run the test suite against that
# instrumented library, and emit BOTH C and Python coverage reports -- entirely
# locally, no third-party service.
#
# Usage:
#   scripts/coverage.sh                 # normal dataset, full suite
#   scripts/coverage.sh --dataset small # forward any pytest args
#   GCOV='llvm-cov gcov' scripts/coverage.sh   # override the gcov tool
#
# Outputs (under the repo root):
#   htmlcov/index.html      Python line-by-line report   (pytest-cov)
#   coverage.xml            Python Cobertura XML
#   htmlcov-c/index.html    C line-by-line report        (gcovr)
#   coverage-c.xml          C Cobertura XML
#
# Requires: cmake, a C compiler with coverage support (gcc or clang), and the
# `coverage` extra (pytest-cov + gcovr + genbadge, plus `test`):
#   uv sync --extra coverage
set -euo pipefail

# Repo root = parent of this script's dir, regardless of CWD.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

# A dedicated build dir so the instrumented objects never clobber the normal
# dev `build/`. It must persist across the pytest run: .gcno is written here at
# compile time, .gcda at process exit -- gcov needs both.
BUILD_DIR="build-coverage"

# Shared-library filename by platform (matches _native_core._candidate_filenames).
case "$(uname -s)" in
  Darwin) LIB_NAME="libevutils_native.dylib" ;;
  *)      LIB_NAME="libevutils_native.so" ;;
esac

# gcovr needs a gcov that matches the compiler. Default `gcov`; on clang/macOS
# use `llvm-cov gcov`. Override with the GCOV env var.
if [ -n "${GCOV:-}" ]; then
  GCOV_ARG=(--gcov-executable "${GCOV}")
elif [ "$(uname -s)" = "Darwin" ]; then
  GCOV_ARG=(--gcov-executable "llvm-cov gcov")
else
  GCOV_ARG=()
fi

echo ">> Configuring instrumented build in $BUILD_DIR/"
cmake -B "$BUILD_DIR" -DEVUTILS_COVERAGE=ON -DCMAKE_BUILD_TYPE=Debug >/dev/null
echo ">> Building"
cmake --build "$BUILD_DIR" -j >/dev/null

# Locate the freshly built instrumented library.
LIB_PATH="$(find "$BUILD_DIR" -name "$LIB_NAME" -print -quit)"
if [ -z "$LIB_PATH" ]; then
  echo "ERROR: instrumented library ($LIB_NAME) not found under $BUILD_DIR/" >&2
  exit 1
fi
LIB_PATH="$(cd "$(dirname "$LIB_PATH")" && pwd)/$(basename "$LIB_PATH")"
echo ">> Instrumented library: $LIB_PATH"

# Clear any stale .gcda from a previous run so counts start clean.
find "$BUILD_DIR" -name '*.gcda' -delete

# The dev-mode loader (src/evutils/io/_native_core.py) would otherwise prefer an
# installed .so; EVUTILS_NATIVE_LIB pins the instrumented one unambiguously.
# --cov* -> Python coverage; the same run also drives the C instrumentation.
echo ">> Running tests"
EVUTILS_NATIVE_LIB="$LIB_PATH" \
  uv run pytest "$@" \
    --cov --cov-report=term-missing --cov-report=xml --cov-report=html

echo ">> Collecting C coverage with gcovr"
mkdir -p htmlcov-c   # gcovr --html-details writes into, but won't create, this dir
uv run gcovr \
  --root . \
  --filter 'csrc/' \
  "${GCOV_ARG[@]}" \
  --txt \
  --xml coverage-c.xml \
  --html-details htmlcov-c/index.html

echo
echo "Reports:"
echo "  Python: htmlcov/index.html   coverage.xml"
echo "  C:      htmlcov-c/index.html  coverage-c.xml"
