# FHE Engine

A Python SDK for GPU-accelerated CKKS homomorphic encryption inference.

---

## Overview

FHE Engine provides a high-level Python interface for running neural network
inference directly on encrypted data. It sits on top of
[FIDESlib](https://github.com/UNIZAR-30226-2025-02/FIDESlib) (a CUDA-accelerated
CKKS library) and [OpenFHE](https://openfhe.org/), exposing encrypted tensor
types and neural network module primitives that require no knowledge of the
underlying cryptographic machinery.

The API is deliberately designed to be familiar to PyTorch users. Module names,
constructor signatures, and the `forward()` calling convention all mirror
`torch.nn` conventions, so that practitioners can map their existing plaintext
models onto encrypted equivalents with minimal conceptual overhead.

---

## Problem

Machine learning models routinely process sensitive inputs: medical records,
financial transactions, biometric data. Conventional deployment forces a choice
between utility and privacy — either the user sends raw data to the server, or
the server sends its model to the user. Neither is acceptable in high-assurance
settings.

Fully Homomorphic Encryption (FHE) removes this trade-off by allowing a server
to compute on ciphertexts it cannot read. However, existing tooling has
significant gaps:

| Library       | Scheme | Compute target | Gap                                           |
|---------------|--------|----------------|-----------------------------------------------|
| TenSEAL       | CKKS   | CPU (OpenMP)   | No GPU support; throughput limited for inference at scale |
| SEAL          | CKKS   | CPU            | Low-level C++ API; no Python layer abstractions |
| HElib         | CKKS / BGV | CPU        | Research-grade API; limited Python support    |
| Concrete ML   | TFHE   | GPU            | TFHE requires boolean/integer ops; floating-point networks need heavy quantization with accuracy loss |
| FIDESlib      | CKKS   | GPU (CUDA)     | C++/CUDA library with Pybind11 bindings; no high-level Python abstractions for tensors or NN modules |

The critical gap: GPU-accelerated CKKS computation (FIDESlib) exists but lacks
the Python-level tensor and module abstractions needed to build real inference
pipelines without writing cryptographic code.

---

## Solution

FHE Engine bridges that gap. It provides:

- **Encrypted tensor and matrix classes** with operator overloading, shape
  tracking, and automatic level (multiplicative depth) accounting.
- **Neural network module primitives** — fully-connected, convolutional,
  batch normalisation, pooling, and polynomial activations — each of which
  knows how to pack its weights, which rotation keys it needs, and how many
  multiplicative levels it consumes.
- **A key management and weight encoding subsystem** that handles all
  cryptographic preparation before inference begins.
- **A fluent builder API** for creating a properly parameterized `CryptoContext`
  without manually configuring OpenFHE.

The user writes Python. FHE Engine handles CKKS.

---

## Architecture

FHE Engine is organized as a three-layer stack:

```
+----------------------------------------------------------+
|  User application (Python)                               |
|  encrypted_output = module.forward(encrypted_input)      |
+----------------------------------------------------------+
|  FHE Engine SDK  (this package)                          |
|  CKKSParams, CryptoContext, EncryptedTensor,             |
|  PackingLayout, WeightEncoder, Module protocol,          |
|  Linear, Conv2d, PolyActivation, Sequential, ...         |
+----------------------------------------------------------+
|  OpenFHE-Python + Pybind11  (binding layer)              |
|  openfhe.CryptoContext, openfhe.Ciphertext, ...          |
+----------------------------------------------------------+
|  FIDESlib  (C++/CUDA compute layer)                      |
|  GPU ciphertext arithmetic, NTT, key-switching           |
+----------------------------------------------------------+
```

### Layer 1 — Python SDK (user-facing)

This is the package you import. It exposes:

- **`CKKSParams`** and **`ContextBuilder`** — scheme parameterization.
- **`CryptoContext`** — the single object that owns the OpenFHE context and the
  FIDESlib GPU handle. All cryptographic operations route through it.
- **`EncryptedTensor`** and **`PlaintextTensor`** — the data types that flow
  between modules.
- **`PackingLayout`** — describes how a logical tensor (e.g., a 1-D activation
  vector of length 512) maps onto CKKS slots.
- **Module protocol and implementations** — `Linear`, `Conv2d`, `PolyActivation`,
  `Sequential`, etc.
- **`WeightEncoder`** and **`KeyManager`** — pre-inference setup utilities.
- **`DepthCost`** and **`RotationSet`** — static analysis types used during
  key planning.

### Layer 2 — OpenFHE-Python / Pybind11

OpenFHE provides the serialization, key generation, and polynomial arithmetic
primitives. FIDESlib wraps those primitives with CUDA-accelerated kernels
exposed via Pybind11. The SDK imports from both and hides the distinction from
the caller.

### Layer 3 — FIDESlib (C++/CUDA)

FIDESlib implements GPU-resident CKKS ciphertext arithmetic: NTT, modular
reduction, key-switching, and rotation. It is the performance-critical
substrate. The SDK delegates all heavy lifting here through the OpenFHE context
and the FIDESlib handle stored in `CryptoContext`.

---

## Key Design Principle: Introspectable Modules

FHE Engine separates **static analysis** from **execution**. Every module
exposes metadata about its cryptographic cost before any data is encrypted.

The `Module` protocol defines three **static analysis methods** that return
information about an operation without performing any cryptography:

- `depth_cost()` — how many multiplicative levels the module consumes.
- `required_rotations(input_layout)` — which rotation keys must be pre-generated.
- `output_layout(input_layout)` — the `PackingLayout` this module produces.

These methods can be called in sequence over a list of modules before any keys
are generated or any data is encrypted. The resulting `DepthCost` sum determines
`CKKSParams.multiplicative_depth`, and the union of all `RotationSet` objects
is passed to `KeyManager.generate_rotation_keys()`.

This design keeps the SDK open for integration with external tools and
automation pipelines, while ensuring that the caller never needs to reason
about levels, rotations, or packing strategies directly.

---

## Scope

### What FHE Engine supports

- **Inference only.** Weights are plaintext (encoded but not encrypted).
  Only the input activations are encrypted.
- **Supported modules:**
  - Fully-connected (`Linear`) via the Halevi-Shoup diagonal matrix-vector
    algorithm.
  - 2-D convolution (`Conv2d`) via the Im2Col transformation in CKKS slot
    space.
  - Element-wise addition and multiplication (`Add`, `Mul`).
  - Polynomial activation functions (`PolyActivation`) evaluated with the
    baby-step/giant-step Paterson-Stockmeyer algorithm. Includes ready-made
    subclasses `SquareActivation` and `ChebyshevReLU`.
  - Batch normalisation (`BatchNorm2d`) folded into a scale-and-shift.
  - Global average pooling (`AdaptiveAvgPool2d`) via slot summation.
  - Flatten (`Flatten`).
  - Bootstrapping (`Bootstrap`) for networks that exceed the depth budget.
  - Sequential container (`Sequential`) for chaining modules with automatic
    depth and rotation accumulation.
- **CKKS scheme only** — native floating-point semantics with configurable
  approximation error.
- **GPU execution** via FIDESlib on any CUDA-capable device.

### What FHE Engine does not support

- Training or backpropagation on encrypted data.
- Non-polynomial activations (ReLU, sigmoid) without polynomial approximation.
- Recurrent architectures (RNN, LSTM) — sequential data dependencies conflict
  with the SIMD packing model.
- Multi-party computation or threshold decryption.
- Schemes other than CKKS (BGV, BFV, TFHE).

---

## Target Comparisons

| Metric                  | TenSEAL (CPU CKKS) | Concrete ML (GPU TFHE) | **FHE Engine (GPU CKKS)** |
|-------------------------|--------------------|------------------------|---------------------------|
| Scheme                  | CKKS               | TFHE                   | CKKS                      |
| Compute target          | CPU (multi-core)   | GPU (CUDA)             | GPU (CUDA)                |
| Floating-point native   | Yes                | No (quantized int)     | Yes                       |
| High-level Python API   | Yes                | Yes                    | Yes                       |
| Extensible Module API   | Limited            | Yes (Brevitas)         | Yes (Module protocol)     |
| Bootstrapping support   | No                 | Yes (programmable)     | Yes (Bootstrap)           |
| Throughput target       | Baseline           | ~10-100x CPU CKKS      | Comparable to Concrete ML |

The primary thesis claim is that GPU-accelerated CKKS can match or exceed GPU
TFHE throughput for floating-point inference while preserving the model
accuracy that TFHE's integer quantization sacrifices.

---

## Dependencies

| Package          | Role                                          |
|------------------|-----------------------------------------------|
| `numpy`          | Weight and activation array handling          |
| `openfhe-python` | CKKS scheme, key generation, serialization    |
| FIDESlib (C++/CUDA via Pybind11) | GPU ciphertext arithmetic    |

Development extras (`pip install fhe-engine[dev]`):

| Package         | Role                         |
|-----------------|------------------------------|
| `pytest`        | Test runner                  |
| `pytest-cov`    | Coverage reporting           |
| `mypy`          | Static type checking         |
| `ruff`          | Linting                      |
| `black`         | Code formatting              |
| `hypothesis`    | Property-based testing       |

---

## Installation

```bash
# Clone the repository
git clone <repo-url>
cd thesis/sdk

# Create and activate the virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install in editable mode with development extras
pip install -e ".[dev]"
```

FIDESlib must be compiled separately and its Pybind11 module placed on
`PYTHONPATH`. See the FIDESlib repository for build instructions.
