enable_language(CUDA)
include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

# These policies are related to SWIG python libraries naming
cmake_policy(SET CMP0078 NEW)
cmake_policy(SET CMP0086 NEW)
find_package(SWIG REQUIRED)
include(${SWIG_USE_FILE})

# SWIG requires the Python header
find_package(PythonLibs REQUIRED)
include_directories(${PYTHON_INCLUDE_PATH})
# TODO: Is NumPy required?
# find_package(NumPy REQUIRED)
# include_directories(${NumPy_INCLUDE_DIRS})

# Must tell CMake that SWIG interface files are CXX files
set_source_files_properties(fft.i PROPERTIES CPLUSPLUS ON)
swig_add_library(fft LANGUAGE python SOURCES fft.i fft.cu)
swig_link_libraries(fft cudart cufft)
set_target_properties(fft PROPERTIES CUDA_ARCHITECTURES "60;72;75;80")
install(FILES "${CMAKE_CURRENT_BINARY_DIR}/fft.py" DESTINATION ffttests)
install(TARGETS fft LIBRARY DESTINATION ffttests)