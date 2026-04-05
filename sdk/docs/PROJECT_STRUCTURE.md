# FHE SDK Project Structure

```text
sdk/
├── pyproject.toml              # Package metadata and build config
├── CMakeLists.txt              # Root CMake (builds pybind11 backend)
├── MANIFEST.in                 # Non-Python files for source distributions
│
├── external/                   # External dependencies
│   └── HEonGPU/                # GPU FHE library (git submodule)
│
├── src/
│   ├── backend/                # C++/CUDA pybind11 bindings to HEonGPU
│   │   ├── CMakeLists.txt
│   │   └── bindings.cu         # pybind11 module exposing CKKS API
│   │
│   └── fhe_sdk/                # High-level Python SDK
│       └── __init__.py         # Re-exports backend bindings
│
├── examples/                   # Usage examples
│   └── ckks_square.py          # Encrypt, square on GPU, decrypt
│
├── tests/                      # Test suite (pytest)
│
├── scripts/
│   ├── install_system_deps.sh  # Builds and installs HEonGPU
│   └── run_example.sh          # Runs an example by name
│
└── docs/
    ├── INTRO.md                # Project overview
    └── PROJECT_STRUCTURE.md    # This file
```
