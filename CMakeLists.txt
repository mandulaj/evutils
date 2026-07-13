cmake_minimum_required(VERSION 3.18)
project(evutils_native LANGUAGES C)

# --- Standard / flags -------------------------------------------------------
set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# Opt-in gcov/llvm-cov instrumentation for C coverage. OFF by default so the
# wheel and normal dev builds are unaffected. Turn on for a standalone build
# whose tree is kept around (scripts/coverage.sh):
#   cmake -B build -DEVUTILS_COVERAGE=ON -DCMAKE_BUILD_TYPE=Debug
option(EVUTILS_COVERAGE "Instrument the C sources for gcov coverage" OFF)

# --- Sources ----------------------------------------------------------------
# Compile every .c under csrc/. CONFIGURE_DEPENDS re-globs when files are
file(GLOB EVUTILS_SOURCES CONFIGURE_DEPENDS
     "${CMAKE_CURRENT_SOURCE_DIR}/csrc/*.c")

add_library(evutils_native SHARED ${EVUTILS_SOURCES})

target_include_directories(evutils_native
    PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/csrc/include")

set_target_properties(evutils_native PROPERTIES
    OUTPUT_NAME "evutils_native"
    # Export all symbols on Windows without needing __declspec annotations.
    # (For a curated public ABI later, switch to an explicit export macro.)
    WINDOWS_EXPORT_ALL_SYMBOLS ON)

# --- Version ----------------------------------------------------------------
# Single-source the C version from the Python build. scikit-build-core sets
# SKBUILD_PROJECT_VERSION from [project].version in pyproject.toml, so the
# EVUTILS_VERSION macro (and evutils_version()) always match the wheel and the
# docs -- no hand-editing csrc on every release. Falls back for standalone
# `cmake --build` dev builds that don't go through scikit-build-core.
if(DEFINED SKBUILD_PROJECT_VERSION)
  set(EVUTILS_VERSION_STR "${SKBUILD_PROJECT_VERSION}")
else()
  set(EVUTILS_VERSION_STR "0.0.0+dev")
endif()
target_compile_definitions(evutils_native PRIVATE
    EVUTILS_VERSION="${EVUTILS_VERSION_STR}")

if(MSVC)
  target_compile_options(evutils_native PRIVATE /O2 /W3)
elseif(EVUTILS_COVERAGE)
  # --coverage == -fprofile-arcs -ftest-coverage. -O0 keeps line mapping honest.
  # Emitted .gcno (compile) and .gcda (run) land next to the objects in the
  # build tree, so this build dir must be preserved for gcov to read them.
  target_compile_options(evutils_native PRIVATE -O0 -g --coverage -Wall -Wextra)
  target_link_options(evutils_native PRIVATE --coverage)
else()
  target_compile_options(evutils_native PRIVATE
      -O3 -Wall -Wextra
      $<$<CONFIG:Debug>:-O0 -g>)
endif()

# --- Install ----------------------------------------------------------------
# When built through scikit-build-core (i.e. via `uv build` / pip), drop the
# shared library *inside* the python package directory in the wheel, so the
# ctypes loader finds it next to evutils/__init__.py.
if(DEFINED SKBUILD_PROJECT_NAME)
  install(TARGETS evutils_native
          LIBRARY DESTINATION "${SKBUILD_PROJECT_NAME}/io"
          RUNTIME DESTINATION "${SKBUILD_PROJECT_NAME}/io")
else()
  # Standalone `cmake --build build` leaves the .so in build/ where the
  # loader's dev-mode search also looks.
  install(TARGETS evutils_native
          LIBRARY DESTINATION lib
          RUNTIME DESTINATION bin)
endif()
