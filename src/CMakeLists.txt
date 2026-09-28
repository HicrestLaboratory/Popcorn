# FlashKKM fused kernel: built with CUTLASS 3.x and without RAFT, because RAFT
# brings CUTLASS 2.x headers and defines CUTLASS_NAMESPACE.
add_library(popcorn_flash_kernel STATIC flash/flash_kernel.cu)
target_link_libraries(popcorn_flash_kernel PRIVATE popcorn_cutlass)
target_compile_options(popcorn_flash_kernel PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)
# Linked into libpopcorn, which can be shared or linked into the Python module
set_target_properties(popcorn_flash_kernel PROPERTIES POSITION_INDEPENDENT_CODE ON)
if(POPCORN_FLASH_INDEX64)
  # PUBLIC: flash_kernel.cuh (flash_index_t) is also included by the popcorn sources
  target_compile_definitions(popcorn_flash_kernel PUBLIC POPCORN_FLASH_INDEX64)
endif()

# The Popcorn library. Static unless BUILD_SHARED_LIBS=ON. The public header
# popcorn_api.hpp needs only the C++ standard library, so the dependencies are
# private. Use from another CMake project: add_subdirectory(<popcorn>) and
# target_link_libraries(<target> PRIVATE Popcorn::popcorn).
add_library(popcorn
  kmeans.cu
  cuda_utils.cu
  kernels/argmin.cu
  kernels/centroids.cu
  kernels/distances.cu
  kernels/convergence.cu
  kernels/normalize.cu
  flash/flash_kmeans.cu
  api.cu
  predict.cu
  ncut.cu
  lloyd.cu
  )
add_library(Popcorn::popcorn ALIAS popcorn)
target_link_libraries(popcorn PRIVATE popcorn_deps popcorn_flash_kernel PUBLIC CUDA::cudart)
target_include_directories(popcorn PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
set_target_properties(popcorn PROPERTIES POSITION_INDEPENDENT_CODE ON)
