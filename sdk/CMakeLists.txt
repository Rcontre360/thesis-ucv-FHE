cmake_minimum_required(VERSION 3.25)
project(fhe-sdk LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Auto-build HEonGPU from submodule if not already installed.
# This runs during `pip install .` so the package is fully self-contained.
set(HEONGPU_INSTALL_DIR "${CMAKE_SOURCE_DIR}/build/heongpu")
file(GLOB _heongpu_cmake_dirs "${HEONGPU_INSTALL_DIR}/lib/cmake/HEonGPU*")
if(NOT _heongpu_cmake_dirs)
    message(STATUS "HEonGPU not found at ${HEONGPU_INSTALL_DIR} — building from submodule...")
    execute_process(
        COMMAND bash "${CMAKE_SOURCE_DIR}/scripts/install_system_deps.sh"
        WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
        RESULT_VARIABLE _heongpu_result
    )
    if(NOT _heongpu_result EQUAL 0)
        message(FATAL_ERROR "install_system_deps.sh failed (exit ${_heongpu_result}). "
            "Run it manually and check the output.")
    endif()
endif()
list(APPEND CMAKE_PREFIX_PATH "${HEONGPU_INSTALL_DIR}")

find_package(CUDAToolkit REQUIRED)
find_package(HEonGPU REQUIRED)
find_package(pybind11 REQUIRED)
find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)

add_subdirectory(src/backend)
