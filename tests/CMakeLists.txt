find_package(Catch2 3 REQUIRED)

add_executable(
  unit_kernels
  unit_kernels.cu
  ../src/cuda_utils.cu
  ../src/kernels/argmin.cu
  ../src/kernels/centroids.cu
  ../src/kernels/distances.cu
  )

target_link_libraries(unit_kernels PRIVATE popcorn_deps Catch2::Catch2WithMain)
set_target_properties(unit_kernels PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/bin)
