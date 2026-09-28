# CUTLASS 3.x headers for the FlashKKM kernel. Header-only: nothing is built.
# Set POPCORN_CUTLASS_DIR to an existing CUTLASS 3.x checkout to skip the download.

set(POPCORN_CUTLASS_VERSION 3.5.1)

if(POPCORN_CUTLASS_DIR)
  set(popcorn_cutlass_include ${POPCORN_CUTLASS_DIR}/include)
else()
  include(FetchContent)
  FetchContent_Declare(popcorn_cutlass
    URL https://github.com/NVIDIA/cutlass/archive/refs/tags/v${POPCORN_CUTLASS_VERSION}.tar.gz
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE
    # No CMakeLists.txt in this subdirectory, so CUTLASS is only downloaded, not configured
    SOURCE_SUBDIR do-not-configure)
  FetchContent_MakeAvailable(popcorn_cutlass)
  set(popcorn_cutlass_include ${popcorn_cutlass_SOURCE_DIR}/include)
endif()

if(NOT EXISTS ${popcorn_cutlass_include}/cute/tensor.hpp)
  message(FATAL_ERROR "CUTLASS 3.x headers not found in ${popcorn_cutlass_include}")
endif()

add_library(popcorn_cutlass INTERFACE)
target_include_directories(popcorn_cutlass SYSTEM INTERFACE ${popcorn_cutlass_include})
