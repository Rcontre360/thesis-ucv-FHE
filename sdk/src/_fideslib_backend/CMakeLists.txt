# CMake for the fideslib backend extension
find_package(pybind11 REQUIRED)

# Include directories
target_include_directories(fideslib_python PRIVATE
    ${OpenFHE_INCLUDE}
    ${OpenFHE_INCLUDE}/third-party/include
    ${OpenFHE_INCLUDE}/core
    ${OpenFHE_INCLUDE}/pke
)

# Extension module
pybind11_add_module(_fideslib_backend
    bindings.cpp
    # Add custom kernels and wrappers here as we develop them
)

target_link_libraries(_fideslib_backend PRIVATE 
    fideslib::fideslib
    ${OpenFHE_LIBRARIES}
)

# The install component for scikit-build-core
install(TARGETS _fideslib_backend LIBRARY DESTINATION fhe_engine COMPONENT python_modules)
