cmake_minimum_required(VERSION 3.25)
project(fhe-sdk LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

find_package(CUDAToolkit REQUIRED)
find_package(HEonGPU REQUIRED)
find_package(pybind11 REQUIRED)
find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)

add_subdirectory(src/backend)
