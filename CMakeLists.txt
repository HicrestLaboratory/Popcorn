cmake_minimum_required(VERSION 3.24)

# Build for the GPU on this machine unless the user sets an architecture list
# (for example -DCMAKE_CUDA_ARCHITECTURES=80 on a node without a GPU).
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES native)
endif()

project(Popcorn VERSION 1.0.0 LANGUAGES CXX CUDA)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

option(POPCORN_FETCH_RAFT "Download the RAFT headers if no RAFT installation is found" ON)
option(POPCORN_BUILD_TESTS "Add the unit_kernels target (needs Catch2 3)" OFF)
option(POPCORN_FLASH_INDEX64 "FlashKKM: 64-bit indices (allows n*d and n*k >= 2^31)" OFF)
option(POPCORN_BUILD_PYTHON "Build the Python module popcorn._popcorn (needs Python >= 3.8)" OFF)

# ------------- dependencies --------------#

find_package(CUDAToolkit REQUIRED)
# Tested: CUDA 12.2 and 12.9 work. CUDA 12.4 fails: the cuSPARSE SpMV in the
# distance step causes an illegal memory access from the second iteration on.
if(CUDAToolkit_VERSION VERSION_GREATER_EQUAL 12.4 AND CUDAToolkit_VERSION VERSION_LESS 12.5)
  message(WARNING
    "CUDA ${CUDAToolkit_VERSION}: the cuSPARSE SpMV in the distance step fails with an "
    "illegal memory access. Use CUDA 12.2 or 12.9.")
elseif(NOT (CUDAToolkit_VERSION VERSION_LESS 12.3 OR
            (CUDAToolkit_VERSION VERSION_GREATER_EQUAL 12.9 AND CUDAToolkit_VERSION VERSION_LESS 13)))
  message(WARNING
    "CUDA ${CUDAToolkit_VERSION} is not tested. Popcorn is tested with CUDA 12.2 and 12.9. "
    "CUDA 12.4 is known to fail.")
endif()

# RAFT is used header-only. CMake first looks for an installed RAFT (set
# CMAKE_PREFIX_PATH or raft_ROOT, for example to the conda environment in
# environment.yml). If none is found, the RAFT headers are downloaded.
set(POPCORN_RAFT_VERSION 24.08)
# Only 24.08 is accepted. src/ uses RAFT APIs (for example raft::cluster)
# that are not stable across RAFT releases. raft-config.cmake does not
# support version ranges, so check the upper bound here.
find_package(raft ${POPCORN_RAFT_VERSION} QUIET)
if(raft_FOUND AND raft_VERSION VERSION_GREATER_EQUAL 24.09)
  message(STATUS "Ignoring installed RAFT ${raft_VERSION}: Popcorn needs RAFT ${POPCORN_RAFT_VERSION}")
  set(raft_FOUND FALSE)
endif()
if(raft_FOUND)
  message(STATUS "Using installed RAFT ${raft_VERSION}: ${raft_DIR}")
elseif(POPCORN_FETCH_RAFT)
  message(STATUS "RAFT ${POPCORN_RAFT_VERSION} not found; downloading the headers")
  include(cmake/fetch_raft.cmake)
else()
  message(FATAL_ERROR
    "RAFT ${POPCORN_RAFT_VERSION} not found. Set CMAKE_PREFIX_PATH to a RAFT "
    "installation, or configure with -DPOPCORN_FETCH_RAFT=ON.")
endif()

set(POPCORN_CUTLASS_DIR "" CACHE PATH "Existing CUTLASS 3.x source directory (downloaded if empty)")
include(cmake/fetch_cutlass.cmake)

# ------------- targets --------------#

add_library(popcorn_deps INTERFACE)
target_link_libraries(popcorn_deps INTERFACE raft::raft CUDA::cublas CUDA::cusparse)

add_subdirectory(src)
if(POPCORN_BUILD_PYTHON)
  add_subdirectory(python)
endif()
if(POPCORN_BUILD_TESTS)
  add_subdirectory(tests)
endif()
