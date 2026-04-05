# fhe_sdk

Python library for GPU-accelerated fully homomorphic encryption (FHE) inference over neural networks. Wraps [HEonGPU](https://github.com/Alisah-Ozcan/HEonGPU) (C++/CUDA) through pybind11 bindings. The scheme is always CKKS. Users never interact with raw cryptographic objects.

Full API reference: [`docs/API.md`](docs/API.md)

## Module map

```
fhe_sdk/
  context.py      # FHEContext — crypto setup and key management
  plaintext.py    # Plaintext — encoded (not encrypted) vector
  ciphertext.py   # Ciphertext — encrypted vector with operator overloads
  nn/
    __init__.py
    linear.py     # Linear layer
    activation.py # Square, ApproxReLU, ApproxSigmoid
    sequential.py # Sequential container
```

The CKKS pipeline is: `list[float]` → **encode** → `Plaintext` → **encrypt** → `Ciphertext`. `FHEContext.encode` and `FHEContext.encrypt` expose each step. `FHEContext.encrypt` also accepts a `list[float]` directly as a convenience (encode + encrypt in one call).

The `_backend` bindings (`CKKSContext`, `CKKSOperator`, `CKKSCiphertext`, etc.) are implementation details and are not part of the public API.

---

## Installation

`fhe_sdk` requires the HEonGPU C++/CUDA library to be compiled and the resulting `_backend.so` pybind11 extension to be importable on `PYTHONPATH`.

**Prerequisites:**

- CUDA-capable GPU with CUDA 11.x or later installed
- HEonGPU built (see build instructions below)
- `_backend.so` present in the `fhe_sdk/` package directory or on `PYTHONPATH`
- Python 3.11+

**Build HEonGPU and the backend extension:**

```bash
cd sdk
mkdir -p build && cd build
cmake ..
make -j$(nproc)
```

**Install the Python package:**

```bash
pip install -e sdk/
```

**Verify:**

```python
import fhe_sdk
ctx = fhe_sdk.FHEContext.default()
```

If `_backend.so` is missing or was compiled against a different CUDA version, the import will raise `ImportError` with a message indicating the missing shared library.

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
