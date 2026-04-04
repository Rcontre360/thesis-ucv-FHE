cmake_minimum_required(VERSION 3.25)
project(fhe-engine LANGUAGES CXX CUDA)

# C++ Standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Find dependencies
find_package(CUDAToolkit REQUIRED)
find_package(HEonGPU REQUIRED)
find_package(pybind11 REQUIRED)
find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)

# Include the backend bindings subdirectory
add_subdirectory(src/_heongpu_backend)
