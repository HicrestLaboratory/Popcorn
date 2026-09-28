# Download the RAFT headers and their header-only dependencies (RMM, CCCL,
# spdlog, fmt, cuCollections) with rapids-cmake. Nothing is compiled: the RAFT
# shared library (raft::compiled) is not built.

set(POPCORN_RAPIDS_VERSION ${POPCORN_RAFT_VERSION})

if(NOT EXISTS ${CMAKE_CURRENT_BINARY_DIR}/RAPIDS.cmake)
  file(DOWNLOAD
    https://raw.githubusercontent.com/rapidsai/rapids-cmake/branch-${POPCORN_RAPIDS_VERSION}/RAPIDS.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/RAPIDS.cmake)
endif()
include(${CMAKE_CURRENT_BINARY_DIR}/RAPIDS.cmake)

include(rapids-cmake)
include(rapids-cpm)

rapids_cpm_init()

set(RAFT_NVTX OFF CACHE BOOL "Enable raft nvtx logging" FORCE)
rapids_cpm_find(raft ${POPCORN_RAPIDS_VERSION}.00
  GLOBAL_TARGETS raft::raft
  CPM_ARGS
    GIT_REPOSITORY https://github.com/rapidsai/raft.git
    GIT_TAG        branch-${POPCORN_RAPIDS_VERSION}
    GIT_SHALLOW    TRUE
    SOURCE_SUBDIR  cpp
    OPTIONS
      "BUILD_TESTS OFF"
      "BUILD_PRIMS_BENCH OFF"
      "BUILD_ANN_BENCH OFF"
      "RAFT_COMPILE_LIBRARY OFF")
