# FHE SDK

A Python SDK for GPU-accelerated CKKS homomorphic encryption inference.

## Overview

FHE SDK provides a high-level Python interface for running neural network
inference directly on encrypted data. It sits on top of
[HEonGPU](https://github.com/Alisah-Ozcan/HEonGPU) (a CUDA-accelerated
FHE library), exposing encrypted tensor types and neural network module
primitives that require no knowledge of the underlying cryptographic machinery.

## Quick Start

```bash
# 1. Install system deps and build HEonGPU into build/heongpu
./scripts/install_system_deps.sh

# 2. Build the Python bindings into build/
cmake -S . -B build -DCMAKE_CUDA_ARCHITECTURES=86 -DCMAKE_PREFIX_PATH=build/heongpu
cmake --build build -j$(nproc)

# 3. Run an example
./scripts/run_example.sh ckks_square
```

## Dependencies

| Package   | Role                                    |
|-----------|-----------------------------------------|
| `numpy`   | Weight and activation array handling    |
| HEonGPU   | GPU-accelerated CKKS/BFV/TFHE (C++/CUDA) |
| pybind11  | C++ to Python bindings                  |
