# FHE Engine Project Architecture

This document defines the professional structure for the `fhe-engine` SDK. It follows the modern **`src/` layout** and uses **`scikit-build-core`** to integrate the C++/CUDA backend with the Python package.

## Directory Structure

```text
sdk/
├── pyproject.toml              # Modern package metadata and build-system config
├── CMakeLists.txt              # Root CMake (Orchestrates C++ & Bindings)
├── MANIFEST.in                 # CRITICAL: Ensures C++/CUDA sources are included in the package
│
├── external/                   # UNTOUCHED C++ DEPENDENCIES (Git Submodules)
│   └── FIDESlib/               # The raw FIDESlib repository (Submodule)
│
├── src/                        # SOURCE CODE DIRECTORY
│   │
│   ├── _fideslib_backend/      # INTERNAL C++/CUDA BINDINGS & CUSTOM KERNELS
│   │   ├── CMakeLists.txt      # CMake for the extension and custom CUDA code
│   │   ├── bindings.cpp        # pybind11 definitions (Low-level mapping)
│   │   ├── kernels/            # OUR CUSTOM CUDA/C++ EXTENSIONS
│   │   │   ├── matmul.cu       # Custom optimized FHE Matrix-Vector multiplication
│   │   │   └── matmul.cuh
│   │   └── wrappers/           # Shims that bridge FIDESlib and our SDK
│   │       └── context_shim.cpp
│   │
│   └── fhe_engine/             # HIGH-LEVEL PYTHON SDK (The Public API)
│       ├── __init__.py         # Exposes the user-facing classes
│       ├── core/               # SDK Core Logic (Layouts, Tensors)
│       ├── modules/            # Neural Network Primitives (PyTorch-like)
│       └── utils/              # Key Management & Weight Encoding
│
├── tests/                      # TEST SUITE (pytest)
│   ├── test_backend.py         # Tests for the custom CUDA kernels and bindings
│   └── test_sdk.py             # Tests for high-level NN modules
│
└── scripts/                    # MAINTENANCE & CI/CD SCRIPTS
    └── install_system_deps.sh  # Installs host-level OpenFHE and FIDESlib
```

---

## Architectural Components

### 1. `external/FIDESlib/`
- **Role:** The foundational GPU library. We treat this as **read-only**.
- **Usage:** We link against its headers and libraries but do not modify its source.

### 2. `src/_fideslib_backend/`
- **Role:** This is where we "extend" FIDESlib.
- **Custom Kernels:** Since FIDESlib is a general-purpose library, we will implement SDK-specific logic (like high-performance Im2Col or specialized Matrix-Vector products) in the `kernels/` folder using CUDA.
- **Bindings:** `bindings.cpp` exposes both the original FIDESlib functions and our custom kernels to Python.

### 3. `src/fhe_engine/`
- **Role:** The user-facing SDK. It implements the interface defined in `docs/API.md`.
- **Logic:** Handles the "Two-Phase Inference" (Static analysis vs Execution).

### 4. `MANIFEST.in`
- **Role:** Lists all non-Python files (`.cpp`, `.cu`, `.h`, `.txt`) that must be included when building the package.
- **Importance:** Without this, `pip install` from source would lack the C++/CUDA code needed to compile the backend.

---

## Developer Workflow

### Initial Setup
```bash
# 1. Install system-level C++ dependencies (OpenFHE/FIDESlib)
./scripts/install_system_deps.sh

# 2. Install the SDK and compile the backend
pip install -e .
```

### Extending the Backend
If you add a new CUDA kernel in `src/_fideslib_backend/kernels/`:
1.  Update `src/_fideslib_backend/CMakeLists.txt` to include the new file.
2.  Add the binding in `src/_fideslib_backend/bindings.cpp`.
3.  Run `pip install -e .` again to recompile.
