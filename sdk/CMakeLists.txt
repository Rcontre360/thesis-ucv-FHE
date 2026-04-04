cmake_minimum_required(VERSION 3.25)
project(fhe-engine)

# C++ Standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find dependencies
# Note: we expect these to be installed by scripts/install_system_deps.sh 
# and their paths to be in CMAKE_PREFIX_PATH
find_package(OpenFHE REQUIRED)
find_package(fideslib REQUIRED)
find_package(pybind11 REQUIRED)

# Python detection (provided by scikit-build-core)
find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)

# Include the backend bindings subdirectory
add_subdirectory(src/_fideslib_backend)
