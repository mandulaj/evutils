cmake_minimum_required(VERSION 3.18)
project(evutils_native LANGUAGES C)

# --- Standard / flags -------------------------------------------------------
set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

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

if(MSVC)
  target_compile_options(evutils_native PRIVATE /O2 /W3)
else()
  target_compile_options(evutils_native PRIVATE
      -O3 -march=native -Wall -Wextra
      $<$<CONFIG:Debug>:-O0 -g>)
endif()

# --- Install ----------------------------------------------------------------
# When built through scikit-build-core (i.e. via `uv build` / pip), drop the
# shared library *inside* the python package directory in the wheel, so the
# ctypes loader finds it next to evutils/__init__.py.
if(DEFINED SKBUILD_PROJECT_NAME)
  install(TARGETS evutils_native
          LIBRARY DESTINATION "${SKBUILD_PROJECT_NAME}"
          RUNTIME DESTINATION "${SKBUILD_PROJECT_NAME}")
else()
  # Standalone `cmake --build build` leaves the .so in build/ where the
  # loader's dev-mode search also looks.
  install(TARGETS evutils_native
          LIBRARY DESTINATION lib
          RUNTIME DESTINATION bin)
endif()
