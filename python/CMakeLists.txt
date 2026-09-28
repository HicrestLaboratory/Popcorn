# Python module popcorn._popcorn (pybind11). pybind11 is taken from an
# installation if found (pip install pybind11), else downloaded (header-only).
find_package(Python 3.8 REQUIRED COMPONENTS Interpreter Development.Module)

find_package(pybind11 CONFIG QUIET)
if(NOT pybind11_FOUND)
  message(STATUS "pybind11 not found; downloading it")
  include(FetchContent)
  FetchContent_Declare(pybind11
    URL https://github.com/pybind/pybind11/archive/refs/tags/v2.13.6.tar.gz
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE)
  FetchContent_MakeAvailable(pybind11)
endif()

pybind11_add_module(_popcorn src/bindings.cpp)
target_link_libraries(_popcorn PRIVATE popcorn)

# Package layout in the build tree, so that PYTHONPATH=<build>/python works
# without installing: <build>/python/popcorn/{__init__.py, _popcorn.so}
set_target_properties(_popcorn PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/popcorn)
configure_file(popcorn/__init__.py ${CMAKE_CURRENT_BINARY_DIR}/popcorn/__init__.py COPYONLY)

# pip install . (scikit-build-core) installs the module into the package
install(TARGETS _popcorn LIBRARY DESTINATION popcorn)
