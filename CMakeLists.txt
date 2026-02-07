
cmake_minimum_required(VERSION 3.10)

# Set policy for CUDA architectures
if(POLICY CMP0104)
  cmake_policy(SET CMP0104 NEW)
endif()

# Set CUDA architectures if using policy CMP0104 NEW
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES all-major)
endif()

project(ARFI_Beamforming LANGUAGES CXX CUDA)

# Use C++17 standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Find HDF5 library via pkg-config (to avoid broken Nix wrappers)
find_package(PkgConfig REQUIRED)
pkg_check_modules(HDF5 REQUIRED hdf5_cpp)
pkg_check_modules(FFTW3 REQUIRED fftw3f)
find_package(OpenMP REQUIRED)

# Add executable
add_executable(arfi_beamformer 
    main.cpp 
    beamformer.cpp 
    beamformer_cuda.cu
    beamformer_cuda_opt.cu
    io.cpp 
    hilbert.cpp
)

# Include directories and link libraries
target_include_directories(arfi_beamformer PRIVATE ${HDF5_INCLUDE_DIRS} ${FFTW3_INCLUDE_DIRS})
target_link_libraries(arfi_beamformer PRIVATE ${HDF5_LIBRARIES} ${FFTW3_LIBRARIES} OpenMP::OpenMP_CXX)
