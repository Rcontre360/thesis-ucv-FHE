# fhe-sdk

Python library for GPU-accelerated fully homomorphic encryption (FHE) inference over neural networks. Wraps [HEonGPU](https://github.com/Alisah-Ozcan/HEonGPU) (C++/CUDA) via pybind11. Scheme is always CKKS; users never interact with raw cryptographic objects.

Two public packages: **`api`** (high-level — `FHEContext`, `Sequential`, `Linear`, `Conv2D`, `ReLU`, `Square`) and **`core`** (enums, errors, base `Layer` ABC). The `_backend` extension is an implementation detail and not part of the public API.

Full API reference: [`docs/API.md`](docs/API.md).

---

## System requirements

This is a CUDA library — every install runs `nvcc`. The CI / development target is the stack below; older or newer versions may work but are not tested.

| Component | Required version | Notes |
| --- | --- | --- |
| NVIDIA GPU | Compute capability ≥ 7.0 (Volta or newer) | Pascal and older lack the tensor-core primitives HEonGPU uses. |
| NVIDIA driver | ≥ 535 | Whatever ships with CUDA 12.8. |
| **CUDA Toolkit** | **12.8** | `nvcc --version` must report 12.8. Set `CUDA_HOME` so `find_package(CUDAToolkit)` resolves. |
| Python | 3.11 or 3.12 | 3.13 untested; 3.10 and below not supported. |
| CMake | ≥ 3.30 | HEonGPU's requirement. `pip install cmake` works if your distro ships an older one. |
| GCC / G++ | 11–13 | Must be CUDA-12.8 compatible. GCC 14 is rejected by `nvcc`. |
| **GMP** | development headers | HEonGPU links against GMP for big-integer math. Install via `gmp-devel` (RHEL/Amazon Linux) or `libgmp-dev` (Debian/Ubuntu). |
| **NTL** | development headers | HEonGPU uses NTL (Number Theory Library) for CKKS cosine approximation. Install via `ntl-devel` (RHEL/Amazon Linux) or `libntl-dev` (Debian/Ubuntu). Requires GMP. |
| **ZLIB** | development headers | HEonGPU's rapids-cmake helper links against ZLIB. Install via `zlib-devel` (RHEL/Amazon Linux) or `zlib1g-dev` (Debian/Ubuntu). |
| **OpenSSL** | development headers | HEonGPU's RNG layer links against libssl/libcrypto. Install via `openssl-devel` (RHEL/Amazon Linux) or `libssl-dev` (Debian/Ubuntu). |
| Ninja | optional | Speeds up the build (`pip install ninja`); CMake falls back to Make otherwise. |
| Git | any recent | Needed to clone the HEonGPU submodule at install time. |
| Disk | ~2 GB free | HEonGPU + bindings build artifacts. |

OS support: **Linux only.** Tested on Ubuntu 22.04 / 24.04. WSL2 works if CUDA is correctly forwarded. macOS and Windows native are not supported (HEonGPU is CUDA-only).

---

## Installation

### From PyPI

```bash
pip install fhe-sdk
```

This downloads the source distribution and triggers a local CMake + CUDA build (≈ 8–15 minutes on a modern desktop). HEonGPU is cloned and built as part of the install.

### From source (development)

```bash
git clone --recurse-submodules https://github.com/Rcontre360/thesis-ucv-FHE.git
cd thesis-ucv-FHE/sdk
pip install -e .
```

If you forgot `--recurse-submodules`:

```bash
git submodule update --init --recursive
```

### Verify

```python
from api import FHEContext
from core.enums import SecurityLevel

ctx = (
    FHEContext()
    .set_poly_modulus_degree(8192)
    .set_coeff_modulus_bit_sizes([60, 40, 40, 60])
    .set_scale(2**40)
    .set_security_level(SecurityLevel.SEC128)
    .build()
)
print("OK")
```

If `from api import FHEContext` prints `OK`, you're good.

---

## Troubleshooting

### `GMP not found` / `NTL/RR.h: No such file or directory` / `Could NOT find ZLIB` / `Could NOT find OpenSSL`

HEonGPU links against GMP, NTL, ZLIB, and OpenSSL. None of these are bundled with CUDA or Python — install them via the system package manager. Install all four in one go to avoid hitting the errors one at a time:

```bash
# RHEL / Amazon Linux / Fedora
sudo yum install -y gmp-devel ntl-devel zlib-devel openssl-devel

# Debian / Ubuntu
sudo apt install -y libgmp-dev libntl-dev zlib1g-dev libssl-dev
```

Then re-run `pip install fhe-sdk`.

If you don't have sudo (e.g. some managed notebook environments), install via conda:

```bash
conda install -y -c conda-forge gmp ntl zlib openssl
export GMP_ROOT=$CONDA_PREFIX
export NTL_ROOT=$CONDA_PREFIX
export ZLIB_ROOT=$CONDA_PREFIX
export OPENSSL_ROOT_DIR=$CONDA_PREFIX
pip install fhe-sdk
```

### `ERROR: Could not find a version that satisfies the requirement fhe-sdk`

Your Python interpreter is older than 3.11. Check with `python --version`. Upgrade Python (or switch conda envs) to 3.11 or 3.12.

### `cmake X.Y.Z is too old. HEonGPU requires cmake >= 3.30`

Your distro/container ships an older CMake. Install a newer one via pip — it lands in your env's `bin` dir which takes precedence over `/usr/bin/cmake`:

```bash
pip install -U cmake
hash -r                 # refresh the shell's PATH cache
cmake --version         # confirm >= 3.30
pip install fhe-sdk
```

### `ImportError: libheongpu.so` (or similar) at runtime

The install completed but the runtime can't locate the C++ library — usually because `CUDA_HOME` / `LD_LIBRARY_PATH` is unset. Re-run the install in a fresh terminal after sourcing your CUDA env, e.g.:

```bash
export CUDA_HOME=/usr/local/cuda-12.8
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
```

### `concrete.compiler.check_gpu_enabled()` returns `False` after installing concrete-ml

When you install `concrete-ml`, it pulls the CPU build of `concrete-python` from PyPI (`Version: 2.10.0`). The GPU build is hosted separately at `https://pypi.zama.ai/gpu` under calendar versioning (`Version: 2024.12.19` or similar). Pip's resolver doesn't reliably prefer the GPU build even with `--extra-index-url` or `--index-url` — both indices host wheels named `concrete-python`, and pip can pick the wrong one based on its internal version comparison heuristics.

**Fix — pin the exact GPU wheel version**, leaving no room for pip to silently substitute:

```bash
pip uninstall -y concrete-python
pip install --no-deps --extra-index-url https://pypi.zama.ai/gpu --trusted-host pypi.zama.ai 'concrete-python==2024.12.19'
```

Replace `2024.12.19` with whatever the current GPU build version is on Zama's index (visit `https://pypi.zama.ai/gpu/concrete-python/` in a browser to check).

`--no-deps` is critical: without it, pip may re-resolve other packages (notably `torch`) and break unrelated parts of your environment. After installing, **restart the kernel** (compiled `.so` files don't reload via `pip install` alone) and verify:

```python
import concrete.compiler
print(concrete.compiler.check_gpu_enabled())    # True
print(concrete.compiler.check_gpu_available())  # True
```

If both print `True`, the GPU build is correctly loaded. You can confirm the installed version on disk with `pip show concrete-python | head -3` — `Version: 2024.12.19` (or newer calendar version) is the GPU build; `Version: 2.10.0` (semver) is the CPU build.

### `nvcc fatal: Unsupported gnu version!`

You're on GCC 14 (or newer). CUDA 12.8 only supports GCC 11–13. Install GCC 12 alongside and point `CC` / `CXX` at it before installing:

```bash
sudo apt install -y gcc-12 g++-12       # Debian/Ubuntu
export CC=gcc-12 CXX=g++-12
pip install fhe-sdk
```

---

## Quickstart

### Crypto primitives

```python
from fhe_sdk import FHEContext
from fhe_sdk.enums import SecurityLevel

# Build a context (fluent setter API)
ctx = (
    FHEContext()
    .set_poly_modulus_degree(8192)
    .set_coeff_modulus_bit_sizes([60, 40, 40, 60])
    .set_scale(2**40)
    .set_security_level(SecurityLevel.SEC128)
    .build()
)

# Or use the one-call default (equivalent to the above)
ctx = FHEContext.default()

# Encode only — produces a Plaintext (not encrypted)
values: list[float] = [0.1, 0.5, -0.3, 0.9]
pt = ctx.encode(values)

# Encrypt — accepts list[float] or Plaintext
ct = ctx.encrypt(values)   # encode + encrypt in one step
ct = ctx.encrypt(pt)       # encrypt an already-encoded Plaintext

# Arithmetic on Ciphertext
a = ctx.encrypt([1.0, 2.0, 3.0, 4.0])
b = ctx.encrypt([0.5, 0.5, 0.5, 0.5])
pt_b = ctx.encode([0.5, 0.5, 0.5, 0.5])

c = a + b                      # ct + ct — homomorphic addition, free
d = a * b                      # ct * ct — homomorphic multiply, consumes one level
e = a + pt_b                   # ct + Plaintext — free
f = a * pt_b                   # ct * Plaintext — consumes one level
g = a * 2.0                    # ct * scalar
h = a + [1.0, 2.0, 3.0, 4.0]  # ct + list[float] — auto-encodes on the fly

# Decrypt
result: list[float] = ctx.decrypt(d)
result: list[float] = d.decrypt()   # shorthand
```

### Neural network inference

This example loads a pre-trained PyTorch model and runs encrypted inference.

```python
import torch
import torch.nn as torch_nn

from fhe_sdk import FHEContext
from fhe_sdk.nn import Sequential, Linear, Square

# Pre-trained PyTorch model
class SmallNet(torch_nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = torch_nn.Linear(64, 64)
        self.fc2 = torch_nn.Linear(64, 10)

    def forward(self, x):
        return self.fc2(self.fc1(x) ** 2)

torch_model = SmallNet()
# torch_model.load_state_dict(torch.load("weights.pt"))

# FHE context — default gives 2 usable levels, enough for one Square activation
ctx = FHEContext.default()

# Build the FHE model
fhe_model = Sequential(
    Linear(in_features=64, out_features=64),
    Square(),
    Linear(in_features=64, out_features=10),
)

# Load weights from the pre-trained PyTorch layers
fhe_model[0].load_from_torch(torch_model.fc1)
fhe_model[2].load_from_torch(torch_model.fc2)

# Encrypted inference
plaintext_input: list[float] = [0.0] * 64   # replace with real features

enc_input = ctx.encrypt(plaintext_input)
enc_output = fhe_model(enc_input)            # Ciphertext of size 10
result: list[float] = enc_output.decrypt()  # list[float] of length 10
```

`fhe_model(enc_input)` is identical to `fhe_model.forward(enc_input)`.

### Loading weights from NumPy

```python
import numpy as np
from fhe_sdk.nn import Linear

W = np.random.randn(10, 64).astype(np.float64)
b = np.random.randn(10).astype(np.float64)

layer = Linear(in_features=64, out_features=10)
layer.load_weights(W, b)   # accepts numpy.ndarray, torch.Tensor, or list[list[float]]
```

---

## Depth Budget Guide

CKKS represents ciphertexts relative to a chain of modulus primes (`q_0 * q_1 * ... * q_L`). Each multiplication consumes one prime. When all intermediate primes are exhausted, no further multiplications are possible.

```
usable_levels = len(coeff_modulus_bit_sizes) - 2
```

The first and last primes are special and not consumed by computations — a chain of length 4 gives 2 usable levels.

### Level cost per layer type

| Layer / Operation | Levels consumed |
|---|---|
| `Linear` | 0 — matrix-vector multiply uses plaintext weights |
| `Square` | 1 |
| `ApproxReLU(degree=3)` | 2 |
| `ApproxReLU(degree=5)` | 3 |
| `ApproxSigmoid(degree=3)` | 2 |
| `ApproxSigmoid(degree=5)` | 3 |

Add 1 extra level as a safety margin.

### Example: 2-layer network with Square

```
Sequential(Linear(64,64), Square(), Linear(64,10))
```

Levels needed: 1. With margin: 2. Chain length: 4.

```python
ctx = (
    FHEContext()
    .set_poly_modulus_degree(8192)
    .set_coeff_modulus_bit_sizes([60, 40, 40, 60])  # 2 usable levels
    .set_scale(2**40)
    .build()
)
# Equivalent: FHEContext.default()
```

### Example: deeper network with ApproxReLU

```
Sequential(
    Linear(128, 64), ApproxReLU(degree=3),   # 2 levels
    Linear(64, 64),  ApproxReLU(degree=3),   # 2 levels
    Linear(64, 10),
)
```

Levels needed: 4. With margin: 5. Chain length: 7. Requires `poly_modulus_degree=16384` to keep the larger modulus sum within the 128-bit security bound.

```python
ctx = (
    FHEContext()
    .set_poly_modulus_degree(16384)
    .set_coeff_modulus_bit_sizes([60, 40, 40, 40, 40, 40, 60])  # 5 usable levels
    .set_scale(2**40)
    .build()
)
```

---

## Crypto Parameter Reference

| `poly_modulus_degree` | `coeff_modulus_bit_sizes` | Total bits | Usable levels | Max slots | Notes |
|---|---|---|---|---|---|
| `4096` | `[40, 20, 40]` | 100 | 1 | 2048 | Minimal. Linear models only. |
| `8192` | `[60, 40, 40, 60]` | 200 | 2 | 4096 | **Default.** One Square or one ApproxReLU(degree=3) + margin. |
| `8192` | `[60, 40, 40, 40, 60]` | 240 | 3 | 4096 | One ApproxReLU(degree=3) + one Square, with margin. |
| `16384` | `[60, 40, 40, 40, 40, 40, 60]` | 320 | 5 | 8192 | Two ApproxReLU(degree=3) activations, with margin. |
| `16384` | `[60, 50, 50, 50, 50, 50, 50, 60]` | 420 | 6 | 8192 | Deeper models or degree-5 activations. |
| `32768` | `[60, 40, 40, 40, 40, 40, 40, 40, 40, 60]` | 540 | 8 | 16384 | Deep models; large input vectors. |

- `poly_modulus_degree` must be a power of 2.
- Max encrypted vector length is `poly_modulus_degree / 2`.
- Interior primes should have bit size equal to `log2(scale)`. Mismatches cause growing noise after rescaling.
- Exceeding the security-level bit-sum limit causes `build()` to raise `ValueError`.
- Use `SecurityLevel.SEC192` or `SEC256` for tighter security; this requires a larger `poly_modulus_degree` or shorter chain.
