#!/bin/bash
# Build and run the unit tests. Catch2 3 must be findable by CMake
# (for example, add its install prefix to CMAKE_PREFIX_PATH).
set -e

PREFIX_ARGS=""
if [ -n "$CONDA_PREFIX" ]; then
  PREFIX_ARGS="-DCMAKE_PREFIX_PATH=$CONDA_PREFIX"
fi

cmake -S . -B build $PREFIX_ARGS -DPOPCORN_BUILD_TESTS=ON "$@"
cmake --build build -j --target unit_kernels
./build/tests/bin/unit_kernels -v high
