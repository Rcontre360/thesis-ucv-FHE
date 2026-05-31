cmake_minimum_required(VERSION 3.25)
project(fhe-sdk LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

find_package(CUDAToolkit REQUIRED)

# Auto-build HEonGPU from external/ if not already installed.
set(HEONGPU_INSTALL_DIR "${CMAKE_SOURCE_DIR}/build/heongpu")
file(GLOB _heongpu_cmake_dirs "${HEONGPU_INSTALL_DIR}/lib/cmake/HEonGPU*")
if(NOT _heongpu_cmake_dirs)
    message(STATUS "HEonGPU not found at ${HEONGPU_INSTALL_DIR} — building from external/HEonGPU...")

    # 1. Ensure submodule source is present (no-op when shipped via pip sdist).
    if(NOT EXISTS "${CMAKE_SOURCE_DIR}/external/HEonGPU/src")
        if(EXISTS "${CMAKE_SOURCE_DIR}/.git")
            execute_process(
                COMMAND git submodule update --init --recursive
                WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
                RESULT_VARIABLE _sub_result
            )
            if(NOT _sub_result EQUAL 0)
                message(FATAL_ERROR "git submodule update failed (exit ${_sub_result})")
            endif()
        else()
            message(FATAL_ERROR "HEonGPU source missing and no .git available — sdist may be incomplete.")
        endif()
    endif()

    # 2. Neutralize HEonGPU's thirdparty build.sh (which runs `git submodule update`
    #    and fails inside a pip sdist; nested submodules are already populated).
    set(_thirdparty_sh "${CMAKE_SOURCE_DIR}/external/HEonGPU/thirdparty/build.sh")
    if(EXISTS "${CMAKE_SOURCE_DIR}/external/HEonGPU/thirdparty/GPU-FFT/CMakeLists.txt")
        file(WRITE "${_thirdparty_sh}" "#!/usr/bin/env bash\nexit 0\n")
        execute_process(COMMAND chmod +x "${_thirdparty_sh}")
    endif()

    # 3. Configure HEonGPU.
    set(_cuda_arch "$ENV{CUDA_ARCH}")
    if(NOT _cuda_arch)
        set(_cuda_arch "86")
    endif()
    execute_process(
        COMMAND ${CMAKE_COMMAND}
            -S "${CMAKE_SOURCE_DIR}/external/HEonGPU"
            -B "${CMAKE_SOURCE_DIR}/external/HEonGPU/build"
            -DCMAKE_BUILD_TYPE=Release
            -DCMAKE_CUDA_ARCHITECTURES=${_cuda_arch}
            -DCMAKE_INSTALL_PREFIX=${HEONGPU_INSTALL_DIR}
            -DTHRUST_INCLUDE_DIR=${CUDAToolkit_INCLUDE_DIRS}
            "-DCMAKE_CUDA_FLAGS=--pre-include cstdint"
        RESULT_VARIABLE _cfg_result
    )
    if(NOT _cfg_result EQUAL 0)
        message(FATAL_ERROR "HEonGPU configure failed (exit ${_cfg_result})")
    endif()

    # 4. Build + install.
    set(_jobs "$ENV{BUILD_JOBS}")
    if(NOT _jobs)
        cmake_host_system_information(RESULT _jobs QUERY NUMBER_OF_LOGICAL_CORES)
    endif()
    execute_process(
        COMMAND ${CMAKE_COMMAND} --build "${CMAKE_SOURCE_DIR}/external/HEonGPU/build" -j${_jobs}
        RESULT_VARIABLE _build_result
    )
    if(NOT _build_result EQUAL 0)
        message(FATAL_ERROR "HEonGPU build failed (exit ${_build_result})")
    endif()
    execute_process(
        COMMAND ${CMAKE_COMMAND} --install "${CMAKE_SOURCE_DIR}/external/HEonGPU/build"
        RESULT_VARIABLE _inst_result
    )
    if(NOT _inst_result EQUAL 0)
        message(FATAL_ERROR "HEonGPU install failed (exit ${_inst_result})")
    endif()
endif()

list(APPEND CMAKE_PREFIX_PATH "${HEONGPU_INSTALL_DIR}")

find_package(HEonGPU REQUIRED)
find_package(pybind11 REQUIRED)
find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)

add_subdirectory(src/backend)
