#!/bin/bash
# Configure and build the Popcorn library in ./build. Extra arguments go to cmake,
# for example: ./scripts/build.sh -DCMAKE_CUDA_ARCHITECTURES=80
set -e

PREFIX_ARGS=""
if [ -n "$CONDA_PREFIX" ]; then
  PREFIX_ARGS="-DCMAKE_PREFIX_PATH=$CONDA_PREFIX"
fi

cmake -S . -B build $PREFIX_ARGS "$@"
cmake --build build -j
