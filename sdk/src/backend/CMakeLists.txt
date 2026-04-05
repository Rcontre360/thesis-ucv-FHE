pybind11_add_module(_backend
    bindings.cu
    bind_enums.cu
    bind_context.cu
    bind_keys.cu
    bind_data.cu
    bind_crypto.cu
    bind_operator.cu
)

target_link_libraries(_backend PRIVATE
    HEonGPU::heongpu
    CUDA::cudart
)

# Force all object files from CUDA static libraries to be linked
# (needed so CUDA fatbin/device symbols are included in the shared object)
set_property(TARGET _backend APPEND PROPERTY
    LINK_OPTIONS
        "LINKER:--whole-archive,$<TARGET_FILE:HEonGPU::heongpu>,--no-whole-archive"
        "LINKER:--whole-archive,$<TARGET_FILE:GPUNTT::ntt>,--no-whole-archive"
        "LINKER:--whole-archive,$<TARGET_FILE:GPUFFT::fft>,--no-whole-archive"
        "LINKER:--whole-archive,$<TARGET_FILE:RNGonGPU::rngongpu>,--no-whole-archive"
)

set_target_properties(_backend PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
    POSITION_INDEPENDENT_CODE ON
)

install(TARGETS _backend LIBRARY DESTINATION fhe_sdk COMPONENT python_modules)
