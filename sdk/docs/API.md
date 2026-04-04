# FHE Engine — API Reference

This document is the authoritative reference for every public symbol in the
`fhe_engine` package. It is organized in the order a user would encounter the
types: configuration first, then context and key management, then the tensor
types, then the module system, then error handling.

The module names and `forward()` calling convention mirror `torch.nn`
conventions so that practitioners already familiar with PyTorch can map their
existing plaintext models onto encrypted equivalents with minimal friction.

---

## Quick Example

The following snippet shows the complete lifecycle for a single encrypted
fully-connected layer.

```python
import numpy as np
from fhe_engine import (
    CKKSParams,
    ContextBuilder,
    KeyManager,
    WeightEncoder,
    RotationSet,
    PackingLayout,
    PackingStrategy,
    Linear,
)

# 1. Define scheme parameters
params = CKKSParams(
    poly_modulus_degree=16384,
    multiplicative_depth=4,
    first_mod_size=60,
    scaling_mod_size=40,
    batch_size=8192,
    scaling_technique="FLEXIBLEAUTO",
)

# 2. Build the CryptoContext (GPU-enabled)
context = ContextBuilder().with_params(params).with_gpu(device_id=0).build()

# 3. Determine required rotation keys before generating anything
in_dim, out_dim = 512, 256
weight_matrix = np.random.randn(out_dim, in_dim).astype(np.float32)
bias_vector   = np.random.randn(out_dim).astype(np.float32)

rotation_set = RotationSet.for_diagonal_gemm(in_dim, out_dim)

# 4. Generate all keys
key_manager = KeyManager(context)
key_manager.generate_key_pair()
key_manager.generate_relinearization_key()
key_manager.generate_rotation_keys(rotation_set)

# 5. Encode weights as plaintexts (never encrypted)
encoder = WeightEncoder(context)
input_layout = PackingLayout(
    strategy=PackingStrategy.FLAT,
    logical_shape=(in_dim,),
    active_slots=in_dim,
    num_ciphertexts=1,
    stride=1,
)
weight_diagonals = encoder.encode_diagonal(weight_matrix, level=0)
encoded_bias     = encoder.encode_bias(bias_vector, out_dim, input_layout, level=1)

# 6. Assemble module
layer = Linear(weight_diagonals, encoded_bias, context)

# 7. Encrypt the input
plaintext_input = np.random.randn(in_dim).astype(np.float32)
encrypted_input = context.encrypt(plaintext_input, layout=input_layout)

# 8. Run inference (fully encrypted)
encrypted_output = layer.forward(encrypted_input)

# 9. Decrypt on the client side
result = context.decrypt(encrypted_output)
print(result[:out_dim])
```

---

## Contents

1. [Enums](#1-enums)
2. [Core Configuration](#2-core-configuration)
   - [CKKSParams](#21-ckksparams)
   - [PackingLayout](#22-packinglayout)
   - [DepthCost](#23-depthcost)
   - [RotationSet](#24-rotationset)
3. [Context and Keys](#3-context-and-keys)
   - [ContextBuilder](#31-contextbuilder)
   - [CryptoContext](#32-cryptocontext)
   - [KeyManager](#33-keymanager)
4. [Tensor Types](#4-tensor-types)
   - [EncryptedTensor](#41-encryptedtensor)
   - [PlaintextTensor](#42-plaintexttensor)
5. [Weight Encoding](#5-weight-encoding)
   - [WeightEncoder](#51-weightencoder)
6. [Module System](#6-module-system)
   - [Module Protocol](#61-module-protocol)
   - [Sequential](#62-sequential)
   - [Linear](#63-linear)
   - [Conv2d](#64-conv2d)
   - [Add](#65-add)
   - [Mul](#66-mul)
   - [Flatten](#67-flatten)
   - [PolyActivation](#68-polyactivation)
   - [AdaptiveAvgPool2d](#69-adaptiveavgpool2d)
   - [BatchNorm2d](#610-batchnorm2d)
   - [Bootstrap](#611-bootstrap)
7. [Error Hierarchy](#7-error-hierarchy)
8. [Two-Phase Inference Pattern](#8-two-phase-inference-pattern)

---

## 1. Enums

### `PackingStrategy`

Controls how a logical tensor maps onto CKKS slots.

```python
class PackingStrategy(enum.Enum):
    FLAT     = "FLAT"
    DIAGONAL = "DIAGONAL"
    IM2COL   = "IM2COL"
```

| Value      | Description                                                                 |
|------------|-----------------------------------------------------------------------------|
| `FLAT`     | Elements stored contiguously in slots 0..N-1. Default for 1-D vectors.     |
| `DIAGONAL` | Diagonals of a matrix packed into successive ciphertexts (Halevi-Shoup).    |
| `IM2COL`   | Input patches for a convolution laid out for Im2Col multiplication.         |

---

### `ScalingTechnique`

Maps directly to the OpenFHE `ScalingTechnique` enumeration.

```python
class ScalingTechnique(enum.Enum):
    FIXEDMANUAL   = "FIXEDMANUAL"
    FIXEDAUTO     = "FIXEDAUTO"
    FLEXIBLEAUTO  = "FLEXIBLEAUTO"
```

| Value           | Description                                                                              |
|-----------------|------------------------------------------------------------------------------------------|
| `FIXEDMANUAL`   | Fixed scaling factor; user must call `rescale` explicitly after every multiplication.    |
| `FIXEDAUTO`     | Fixed scaling factor; OpenFHE inserts rescaling automatically.                           |
| `FLEXIBLEAUTO`  | Adaptive scaling factor; minimizes approximation error, recommended for deep networks.   |

---

## 2. Core Configuration

### 2.1 `CKKSParams`

```python
@dataclass(frozen=True)
class CKKSParams:
    poly_modulus_degree: int
    multiplicative_depth: int
    first_mod_size:      int = 60
    scaling_mod_size:    int = 40
    batch_size:          int = 0
    scaling_technique:   ScalingTechnique = ScalingTechnique.FIXEDAUTO
    num_large_digits:    int = 0
```

Immutable CKKS scheme parameters. Passed to `ContextBuilder` once; never
modified after the context is created.

**Attributes**

| Attribute              | Type               | Description                                                                                                        |
|------------------------|--------------------|--------------------------------------------------------------------------------------------------------------------|
| `poly_modulus_degree`  | `int`              | Ring dimension N. Must be a power of 2 (e.g., 8192, 16384, 32768). Slot count equals N / 2.                       |
| `multiplicative_depth` | `int`              | Number of levels L in the modulus chain. Determines how many sequential multiplications are possible before bootstrapping. |
| `first_mod_size`       | `int`              | Bit size of the special first modulus Q_0. Typically 60.                                                           |
| `scaling_mod_size`     | `int`              | Bit size of each remaining modulus prime. Typically 40–50. Affects precision.                                      |
| `batch_size`           | `int`              | Number of slots to activate. Must be <= N / 2. Value 0 means use the full N / 2 slots.                             |
| `scaling_technique`    | `ScalingTechnique` | Rescaling strategy. `FLEXIBLEAUTO` is recommended for inference.                                                   |
| `num_large_digits`     | `int`              | Hybrid key-switching parameter. 0 lets the library choose.                                                         |

**Properties**

| Property     | Return type | Description                                                                      |
|--------------|-------------|----------------------------------------------------------------------------------|
| `slot_count` | `int`       | Effective number of SIMD slots per ciphertext. Returns `batch_size` if set, otherwise `poly_modulus_degree // 2`. |

---

### 2.2 `PackingLayout`

```python
@dataclass(frozen=True)
class PackingLayout:
    strategy:      PackingStrategy
    logical_shape: tuple[int, ...]
    active_slots:  int
    num_ciphertexts: int
    stride:        int
```

Describes exactly how a logical tensor occupies CKKS slots across one or more
ciphertexts. Frozen so it can be safely used as a dictionary key and shared
across threads.

`PackingLayout` carries no cryptographic state. It is a pure description of
geometry. `WeightEncoder` and the module implementations use it to compute slot
indices, required rotations, and output shapes without touching any ciphertext.

**Attributes**

| Attribute        | Type                | Description                                                                              |
|------------------|---------------------|------------------------------------------------------------------------------------------|
| `strategy`       | `PackingStrategy`   | Packing scheme used (FLAT, DIAGONAL, or IM2COL).                                         |
| `logical_shape`  | `tuple[int, ...]`   | Shape of the logical tensor in the usual NumPy sense, e.g. `(512,)` or `(32, 28, 28)`.  |
| `active_slots`   | `int`               | Number of slots that carry meaningful data. Remaining slots are zero-padded.             |
| `num_ciphertexts`| `int`               | Number of ciphertexts required to hold the full tensor under this layout.                |
| `stride`         | `int`               | Step between logically adjacent elements in the slot vector. 1 for FLAT packing.        |

**Methods**

```python
def slot_index(self, *logical_indices: int) -> int:
    ...
```

Map a logical multi-index to its physical slot position within a single
ciphertext. Raises `PackingError` if the index is out of bounds.

---

```python
def required_rotations(self) -> set[int]:
    ...
```

Return the set of rotation steps that an operation consuming this layout must
be able to apply. Used during key-planning traversal before any ciphertexts
exist.

---

```python
def is_compatible_with(self, other: PackingLayout) -> bool:
    ...
```

Return `True` if the output of one module with this layout can be fed directly
into a module expecting `other` without a re-packing step.

---

### 2.3 `DepthCost`

```python
@dataclass(frozen=True)
class DepthCost:
    levels: int
```

A frozen value type representing the number of multiplicative levels consumed
by one operation or a sequence of operations. Used exclusively for static
analysis — no cryptographic operations are performed.

**Attributes**

| Attribute | Type  | Description                                                           |
|-----------|-------|-----------------------------------------------------------------------|
| `levels`  | `int` | Non-negative integer count of levels consumed. 0 means no cost.      |

**Operators**

```python
def __add__(self, other: DepthCost) -> DepthCost:
    ...
```

Return a new `DepthCost` whose `levels` is the sum of both operands.
Used to accumulate cost over a sequence of modules.

**Class methods**

```python
@classmethod
def for_linear(cls) -> DepthCost:
    ...
```

Returns `DepthCost(levels=1)`. A matrix-vector multiplication consumes one
level (one rescaling after the ciphertext-plaintext product).

---

```python
@classmethod
def for_conv2d(cls) -> DepthCost:
    ...
```

Returns `DepthCost(levels=1)`. Im2Col convolution costs the same as a single
matrix-vector multiply.

---

```python
@classmethod
def for_polynomial(cls, degree: int) -> DepthCost:
    ...
```

Returns `DepthCost(levels=ceil(log2(degree)))`. Polynomial evaluation via the
Paterson-Stockmeyer algorithm requires logarithmically many levels in the
polynomial degree.

| `degree` | `levels` |
|----------|----------|
| 2        | 1        |
| 3–4      | 2        |
| 5–8      | 3        |
| 9–16     | 4        |

---

```python
@classmethod
def for_addition(cls) -> DepthCost:
    ...
```

Returns `DepthCost(levels=0)`. Addition does not consume multiplicative levels.

---

```python
@classmethod
def for_bootstrap(cls) -> DepthCost:
    ...
```

Returns `DepthCost(levels=0)`. Bootstrapping refreshes the level budget rather
than consuming it.

---

```python
@classmethod
def zero(cls) -> DepthCost:
    ...
```

Returns `DepthCost(levels=0)`. Convenience alias for operations with no depth
cost (e.g., `Flatten`).

---

### 2.4 `RotationSet`

```python
class RotationSet:
    steps: set[int]
```

Accumulates the complete set of rotation steps that must have pre-computed
evaluation keys before an inference graph can execute. Rotation steps are
signed integers: positive = left-rotate, negative = right-rotate.

A graph compiler collects one `RotationSet` per module via
`Module.required_rotations()`, merges them all, and passes the result to
`KeyManager.generate_rotation_keys()`. This ensures no rotation key is missing
at inference time.

**Attributes**

| Attribute | Type       | Description                                          |
|-----------|------------|------------------------------------------------------|
| `steps`   | `set[int]` | Current accumulated set of rotation step values.     |

**Methods**

```python
def update(self, steps: Iterable[int]) -> None:
    ...
```

Add rotation steps to the set. Duplicates are ignored.

---

```python
def merge(self, other: RotationSet) -> None:
    ...
```

Merge all steps from `other` into this set in place.

---

**Class methods**

```python
@classmethod
def for_diagonal_gemm(cls, in_dim: int, out_dim: int) -> RotationSet:
    ...
```

Return the rotation steps required by the Halevi-Shoup diagonal
matrix-vector algorithm for a matrix of shape `(out_dim, in_dim)`.
The number of steps equals `min(in_dim, out_dim)`.

---

```python
@classmethod
def for_im2col_conv(
    cls,
    kernel_h: int,
    kernel_w: int,
    channels_in: int,
    stride: int,
) -> RotationSet:
    ...
```

Return the rotation steps required by Im2Col convolution for a kernel of
shape `(channels_in, kernel_h, kernel_w)` applied with the given stride.

---

```python
@classmethod
def for_sum_reduction(cls, n: int) -> RotationSet:
    ...
```

Return the `ceil(log2(n))` rotation steps needed to sum `n` adjacent slots
into a single slot (used by dot-product accumulation and global average pool).

---

## 3. Context and Keys

### 3.1 `ContextBuilder`

```python
class ContextBuilder:
    ...
```

A fluent builder for `CryptoContext`. Enforces that all required parameters are
provided before construction. Raises `ContextError` if `build()` is called
without supplying `CKKSParams`.

**Methods**

```python
def with_params(self, params: CKKSParams) -> ContextBuilder:
    ...
```

Set the CKKS scheme parameters. Returns `self` for chaining.

---

```python
def with_gpu(self, device_id: int = 0) -> ContextBuilder:
    ...
```

Enable GPU execution on CUDA device `device_id`. If not called, the context
falls back to CPU execution through OpenFHE. Returns `self` for chaining.

---

```python
def build(self) -> CryptoContext:
    ...
```

Construct and return the `CryptoContext`. Initializes the OpenFHE context and,
if `with_gpu()` was called, the FIDESlib GPU handle. Raises `ContextError` if
`with_params()` was not called first.

**Typical usage**

```python
context = (
    ContextBuilder()
    .with_params(params)
    .with_gpu(device_id=0)
    .build()
)
```

---

### 3.2 `CryptoContext`

```python
class CryptoContext:
    ...
```

The central object of an FHE Engine session. Owns the OpenFHE `CryptoContext`,
the FIDESlib GPU handle, and the set of registered rotation keys. All
ciphertext operations in the SDK ultimately delegate here.

Do not construct directly. Use `ContextBuilder`.

A `CryptoContext` instance is **not thread-safe**. The FIDESlib GPU stream is
single-threaded. Do not share a context across threads without external
synchronization.

**Properties**

| Property               | Type                        | Description                                                              |
|------------------------|-----------------------------|--------------------------------------------------------------------------|
| `params`               | `CKKSParams`                | The immutable scheme parameters supplied at construction time.           |
| `slot_count`           | `int`                       | Convenience alias for `params.slot_count`.                               |
| `registered_rotations` | `frozenset[int]`            | The set of rotation steps for which evaluation keys exist.               |

**Encrypt / Decrypt**

```python
def encrypt(
    self,
    values: np.ndarray,
    layout: PackingLayout,
) -> EncryptedTensor:
    ...
```

Encrypt a NumPy array into a new `EncryptedTensor`. The array is packed into
CKKS slots according to `layout`. Unused slots are zero-padded.

Raises `PackingError` if `values.size > slot_count`.

---

```python
def decrypt(self, tensor: EncryptedTensor) -> np.ndarray:
    ...
```

Decrypt `tensor` and return a NumPy array of length `slot_count`. The caller
uses `tensor.layout` to extract the meaningful elements.

---

**Arithmetic**

These methods are called by `Module` implementations and by `EncryptedTensor`
operator overloads. They are part of the public API because they are needed for
custom module authoring, but typical users work with modules rather than calling
them directly.

```python
def add_ct_ct(
    self,
    a: EncryptedTensor,
    b: EncryptedTensor,
) -> EncryptedTensor:
    ...
```

Homomorphic ciphertext-ciphertext addition. Both operands must share the same
`CryptoContext`. The result inherits the lower level of the two inputs.
Raises `ShapeError` if the layouts are incompatible.

---

```python
def add_ct_pt(
    self,
    ct: EncryptedTensor,
    pt: PlaintextTensor,
) -> EncryptedTensor:
    ...
```

Homomorphic ciphertext-plaintext addition. Does not consume a multiplicative
level.

---

```python
def mul_ct_ct(
    self,
    a: EncryptedTensor,
    b: EncryptedTensor,
) -> EncryptedTensor:
    ...
```

Homomorphic ciphertext-ciphertext multiplication. Consumes one level.
Raises `LevelError` if either operand has no levels remaining.

---

```python
def mul_ct_pt(
    self,
    ct: EncryptedTensor,
    pt: PlaintextTensor,
) -> EncryptedTensor:
    ...
```

Homomorphic ciphertext-plaintext multiplication. Consumes one level.

---

**Rotation**

```python
def rotate(
    self,
    tensor: EncryptedTensor,
    steps: int,
) -> EncryptedTensor:
    ...
```

Left-rotate the slot vector of `tensor` by `steps` positions. Requires that a
rotation key for `steps` was pre-generated.
Raises `KeyMissingError` if the key is absent.

---

```python
def rotate_hoisted(
    self,
    tensor: EncryptedTensor,
    steps: list[int],
) -> list[EncryptedTensor]:
    ...
```

Apply multiple rotations to `tensor` using a single key-switch decomposition.
This is the critical optimization for the diagonal GEMM algorithm: decompose
the ciphertext once, then apply all `len(steps)` rotation keys, reducing the
per-rotation cost from two NTTs to roughly one.

Returns a list of `EncryptedTensor` objects in the same order as `steps`.
Raises `KeyMissingError` if any step in `steps` lacks a key.

---

**Maintenance**

```python
def rescale(self, tensor: EncryptedTensor) -> EncryptedTensor:
    ...
```

Remove the top prime from the modulus chain, reducing the ciphertext size and
re-calibrating the scaling factor. Required after each multiplication when
`scaling_technique` is `FIXEDMANUAL`. Called automatically in `FIXEDAUTO` and
`FLEXIBLEAUTO` modes.

---

```python
def relinearize(self, tensor: EncryptedTensor) -> EncryptedTensor:
    ...
```

Reduce a ciphertext from three polynomial components (produced by
ciphertext-ciphertext multiplication) back to two, using the relinearization
key. Must be called after `mul_ct_ct` in manual mode; called automatically
when a relinearization key is present in auto mode.

---

```python
def bootstrap(self, tensor: EncryptedTensor) -> EncryptedTensor:
    ...
```

Refresh the multiplicative level budget of `tensor`. After bootstrapping,
`tensor.levels_remaining` is restored to approximately
`params.multiplicative_depth`. Requires that bootstrapping keys were generated
via `KeyManager.generate_bootstrapping_keys()`.

Bootstrapping is computationally expensive. Use `Bootstrap` to insert it
at the right point in a network rather than calling this method manually.

---

```python
def level_down(
    self,
    tensor: EncryptedTensor,
    levels: int,
) -> EncryptedTensor:
    ...
```

Manually drop `levels` levels from `tensor` without performing a multiplication.
Used to align two ciphertexts to the same level before addition.

---

### 3.3 `KeyManager`

```python
class KeyManager:
    def __init__(self, context: CryptoContext) -> None:
        ...
```

Handles all key generation for a `CryptoContext`. Keys are stored in the
OpenFHE context and persist for the lifetime of the session.

Key generation must be completed before any encryption or homomorphic
operation is attempted. The recommended order is:

1. `generate_key_pair()`
2. `generate_relinearization_key()`
3. `generate_rotation_keys(rotation_set)`
4. `generate_bootstrapping_keys()` (optional, only for deep networks)

**Methods**

```python
def generate_key_pair(self) -> None:
    ...
```

Generate a public/secret key pair and register it in the context.

---

```python
def generate_relinearization_key(self) -> None:
    ...
```

Generate the evaluation key used by `relinearize()` after ciphertext-ciphertext
multiplication. Must be called after `generate_key_pair()`.

---

```python
def generate_rotation_keys(self, rotation_set: RotationSet) -> None:
    ...
```

Generate one rotation evaluation key per step in `rotation_set.steps`.
This is the most time-consuming key generation step for wide networks.
Collect the complete `RotationSet` from all modules before calling this method
to avoid redundant key generation.

---

```python
def generate_bootstrapping_keys(self) -> None:
    ...
```

Generate the evaluation keys required for `CryptoContext.bootstrap()`.
Only needed if the network depth exceeds `params.multiplicative_depth`.

---

**Properties**

| Property     | Type        | Description                                        |
|--------------|-------------|----------------------------------------------------|
| `public_key` | `PublicKey` | The public key for encrypting inputs.              |
| `secret_key` | `SecretKey` | The secret key for decrypting outputs. Keep safe.  |

---

## 4. Tensor Types

### 4.1 `EncryptedTensor`

```python
class EncryptedTensor:
    ...
```

The primary data type that flows between modules during inference. Wraps one or
more OpenFHE ciphertexts (when `layout.num_ciphertexts > 1`) together with
packing metadata. All arithmetic operators delegate to the owning
`CryptoContext`.

Do not construct directly. Produce instances via `CryptoContext.encrypt()` or
as the output of module `forward()` calls.

**Properties**

| Property           | Type            | Description                                                                                   |
|--------------------|-----------------|-----------------------------------------------------------------------------------------------|
| `context`          | `CryptoContext` | The context that owns and can operate on this tensor.                                         |
| `layout`           | `PackingLayout` | Describes how the logical tensor maps onto slots.                                             |
| `levels_remaining` | `int`           | Number of multiplicative levels still available before bootstrapping is required.             |
| `current_level`    | `int`           | Current position in the modulus chain. 0 = fresh (all levels available).                     |
| `scale`            | `float`         | The current scaling factor. Tracks accumulated scaling across multiplications.                |
| `shape`            | `tuple[int, ...]` | The logical shape of the tensor, equal to `layout.logical_shape`.                           |

**Methods**

```python
def can_apply_op(self, cost: DepthCost) -> bool:
    ...
```

Return `True` if `self.levels_remaining >= cost.levels`. Convenience check used
inside module `forward()` implementations before issuing operations.

---

**Operator overloads**

All operators delegate to `self.context` and return a new `EncryptedTensor`.

```python
def __add__(self, other: EncryptedTensor | PlaintextTensor) -> EncryptedTensor:
    ...

def __mul__(self, other: EncryptedTensor | PlaintextTensor) -> EncryptedTensor:
    ...

def __neg__(self) -> EncryptedTensor:
    ...
```

`__neg__` is implemented as multiplication by the plaintext scalar -1 and
does not consume a level.

---

### 4.2 `PlaintextTensor`

```python
class PlaintextTensor:
    ...
```

An encoded (but not encrypted) tensor representing network weights, biases, or
polynomial coefficients. Produced exclusively by `WeightEncoder`. Users never
construct this class directly.

A `PlaintextTensor` is bound to a specific level in the modulus chain. If the
ciphertext it will be multiplied against is at a different level, the SDK
automatically aligns them.

**Properties**

| Property | Type            | Description                                                                       |
|----------|-----------------|-----------------------------------------------------------------------------------|
| `level`  | `int`           | The modulus-chain level this plaintext is encoded for.                            |
| `scale`  | `float`         | The scaling factor used during encoding.                                          |
| `layout` | `PackingLayout` | Packing layout of the encoded data, matching the ciphertext it is designed for.   |

---

## 5. Weight Encoding

### 5.1 `WeightEncoder`

```python
class WeightEncoder:
    def __init__(self, context: CryptoContext) -> None:
        ...
```

Translates NumPy arrays into `PlaintextTensor` objects ready for homomorphic
multiplication. Each encoding method targets a specific packing strategy.

Encoding is performed once during model setup, not during inference. The
resulting `PlaintextTensor` objects are stored in module instances and reused
across all inference calls.

**Methods**

```python
def encode_flat(
    self,
    values: np.ndarray,
    level: int,
) -> PlaintextTensor:
    ...
```

Encode a 1-D array into consecutive slots using FLAT packing. Suitable for
element-wise operations and bias addition on 1-D activations.

---

```python
def encode_diagonal(
    self,
    matrix: np.ndarray,
    level: int,
) -> list[PlaintextTensor]:
    ...
```

Encode the diagonals of `matrix` for use in the Halevi-Shoup diagonal
matrix-vector algorithm. Returns one `PlaintextTensor` per diagonal.
`matrix` must be 2-D with shape `(out_dim, in_dim)`.

---

```python
def encode_im2col_kernel(
    self,
    kernel: np.ndarray,
    input_shape: tuple[int, ...],
    level: int,
) -> list[PlaintextTensor]:
    ...
```

Encode a convolution kernel in Im2Col format. Returns one `PlaintextTensor`
per patch position. `kernel` has shape `(out_channels, in_channels, kH, kW)`.
`input_shape` is `(in_channels, H, W)`.

---

```python
def encode_bias(
    self,
    bias: np.ndarray,
    output_dim: int,
    layout: PackingLayout,
    level: int,
) -> PlaintextTensor:
    ...
```

Encode a bias vector replicated across the slot layout so it can be added
directly to the output ciphertext of a linear or convolution module. The
replication pattern is derived from `layout`.

---

```python
def encode_polynomial_coefficients(
    self,
    coeffs: list[float],
    level: int,
) -> list[PlaintextTensor]:
    ...
```

Encode the coefficients of a polynomial activation function. Returns one
`PlaintextTensor` per coefficient. Used by `PolyActivation`.

---

## 6. Module System

### 6.1 `Module` Protocol

```python
@runtime_checkable
class Module(Protocol):
    def depth_cost(self) -> DepthCost:
        ...

    def required_rotations(
        self,
        input_layout: PackingLayout,
    ) -> RotationSet:
        ...

    def output_layout(
        self,
        input_layout: PackingLayout,
    ) -> PackingLayout:
        ...

    def forward(
        self,
        x: EncryptedTensor,
    ) -> EncryptedTensor:
        ...
```

The structural type that every module implementation must satisfy. Because it is
a `Protocol`, any class that implements all four methods is automatically
considered a `Module` — no inheritance required. This mirrors the role of
`torch.nn.Module` in PyTorch.

The first three methods perform **static analysis** that requires no
cryptographic state and no ciphertext. They can be called on freshly
constructed module objects before any keys are generated or any data is
encrypted. This property enables graph-level planning of the full depth budget
and rotation key set before inference begins (see
[Two-Phase Inference Pattern](#8-two-phase-inference-pattern)).

Note: `Add` and `Mul` are the only exceptions to the single-input `forward()`
signature; they take two `EncryptedTensor` arguments because they represent
binary operations.

**Methods**

| Method                                        | Cryptographic? | Description                                                                                                                         |
|-----------------------------------------------|----------------|-------------------------------------------------------------------------------------------------------------------------------------|
| `depth_cost() -> DepthCost`                   | No             | Return the number of multiplicative levels this module consumes. Used to set `CKKSParams.multiplicative_depth`.                     |
| `required_rotations(input_layout) -> RotationSet` | No         | Return all rotation steps this module will need at inference time. Used to call `KeyManager.generate_rotation_keys()` once up front.|
| `output_layout(input_layout) -> PackingLayout`| No             | Return the `PackingLayout` this module produces given the input layout. Enables layout propagation through the graph.               |
| `forward(x) -> EncryptedTensor`               | Yes            | Execute the module homomorphically. All listed preconditions (keys, levels) must be satisfied.                                       |

---

### 6.2 `Sequential`

```python
class Sequential:
    def __init__(self, *modules: Module) -> None:
        ...
```

A container that chains modules in order, passing the output of each module as
the input to the next. Mirrors `torch.nn.Sequential`.

`Sequential` implements the `Module` protocol itself, so it can be nested
inside other `Sequential` containers or used anywhere a `Module` is expected.

- `depth_cost()` returns the sum of `depth_cost()` for all contained modules.
- `required_rotations(input_layout)` propagates the layout through each module
  in order, collecting and merging all `RotationSet` objects.
- `output_layout(input_layout)` propagates the layout through each module in
  order, returning the layout produced by the last module.
- `forward(x)` passes `x` through each module in order and returns the final
  output.

**Constructor parameters**

| Parameter  | Type       | Description                                      |
|------------|------------|--------------------------------------------------|
| `*modules` | `Module`   | Ordered sequence of modules to chain.            |

**Typical usage**

```python
from fhe_engine import Sequential, Linear, PolyActivation, Flatten

model = Sequential(
    Flatten(context),
    Linear(weight_diagonals_1, bias_1, context),
    PolyActivation(coeffs_1, degree=4, context=context),
    Linear(weight_diagonals_2, bias_2, context),
)

# Static analysis — no crypto
total_depth = model.depth_cost()
all_rotations = model.required_rotations(input_layout)

# Inference
encrypted_output = model.forward(encrypted_input)
```

---

### 6.3 `Linear`

```python
class Linear:
    def __init__(
        self,
        weight_diagonals: list[PlaintextTensor],
        bias: PlaintextTensor | None,
        context: CryptoContext,
    ) -> None:
        ...
```

Fully-connected (dense) module implemented via the Halevi-Shoup diagonal
matrix-vector multiplication algorithm. Uses `rotate_hoisted` for efficiency.
Mirrors `torch.nn.Linear`.

- `depth_cost()` returns `DepthCost.for_linear()` (1 level).
- `required_rotations()` returns `RotationSet.for_diagonal_gemm(in_dim, out_dim)`.
- Bias addition is free (ciphertext-plaintext, 0 levels).

**Constructor parameters**

| Parameter          | Type                    | Description                                                      |
|--------------------|-------------------------|------------------------------------------------------------------|
| `weight_diagonals` | `list[PlaintextTensor]` | Diagonals from `WeightEncoder.encode_diagonal()`.                |
| `bias`             | `PlaintextTensor | None` | Encoded bias from `WeightEncoder.encode_bias()`, or `None`.     |
| `context`          | `CryptoContext`         | The owning context.                                              |

---

### 6.4 `Conv2d`

```python
class Conv2d:
    def __init__(
        self,
        kernel_diagonals: list[PlaintextTensor],
        bias: PlaintextTensor | None,
        input_shape: tuple[int, ...],
        kernel_shape: tuple[int, ...],
        stride: int,
        padding: str,
        context: CryptoContext,
    ) -> None:
        ...
```

2-D convolutional module implemented via the Im2Col transformation in CKKS slot
space. The input activation map is re-packed as an Im2Col matrix, which
reduces convolution to a matrix-vector multiplication. Mirrors
`torch.nn.Conv2d`.

- `depth_cost()` returns `DepthCost.for_conv2d()` (1 level).
- `required_rotations()` returns `RotationSet.for_im2col_conv(kH, kW, C_in, stride)`.

**Constructor parameters**

| Parameter         | Type                    | Description                                                               |
|-------------------|-------------------------|---------------------------------------------------------------------------|
| `kernel_diagonals`| `list[PlaintextTensor]` | Encoded kernel from `WeightEncoder.encode_im2col_kernel()`.               |
| `bias`            | `PlaintextTensor | None` | Encoded bias, or `None`.                                                 |
| `input_shape`     | `tuple[int, ...]`       | Shape of the input activation map `(C_in, H, W)`.                        |
| `kernel_shape`    | `tuple[int, ...]`       | Shape of the convolution kernel `(C_out, C_in, kH, kW)`.                 |
| `stride`          | `int`                   | Convolution stride (same in both spatial dimensions).                     |
| `padding`         | `str`                   | `"SAME"` for zero-padding to preserve spatial size, `"VALID"` for none.  |
| `context`         | `CryptoContext`         | The owning context.                                                       |

---

### 6.5 `Add`

```python
class Add:
    def __init__(self, context: CryptoContext) -> None:
        ...
```

Element-wise addition of two `EncryptedTensor` operands. Delegates to
`CryptoContext.add_ct_ct()`.

- `depth_cost()` returns `DepthCost.for_addition()` (0 levels).
- `required_rotations()` returns an empty `RotationSet`.
- `forward(a, b)` takes two `EncryptedTensor` arguments (the only module that
  does not follow the single-input convention).

```python
def forward(self, a: EncryptedTensor, b: EncryptedTensor) -> EncryptedTensor:
    ...
```

---

### 6.6 `Mul`

```python
class Mul:
    def __init__(self, context: CryptoContext) -> None:
        ...
```

Element-wise multiplication of two `EncryptedTensor` operands. Delegates to
`CryptoContext.mul_ct_ct()`, followed by automatic relinearization.

- `depth_cost()` returns `DepthCost(levels=1)`.
- `required_rotations()` returns an empty `RotationSet`.
- `forward(a, b)` takes two `EncryptedTensor` arguments (the only module that
  does not follow the single-input convention alongside `Add`).

```python
def forward(self, a: EncryptedTensor, b: EncryptedTensor) -> EncryptedTensor:
    ...
```

---

### 6.7 `Flatten`

```python
class Flatten:
    ...
```

Reshapes the logical tensor to a 1-D vector by updating the `PackingLayout`
metadata. No homomorphic operations are performed; no levels are consumed.
Mirrors `torch.nn.Flatten`.

- `depth_cost()` returns `DepthCost.zero()`.
- `required_rotations()` returns an empty `RotationSet`.
- `output_layout()` returns a FLAT layout with `logical_shape=(n,)` where
  `n = product(input_layout.logical_shape)`.

```python
def forward(self, x: EncryptedTensor) -> EncryptedTensor:
    ...
```

---

### 6.8 `PolyActivation`

```python
class PolyActivation:
    def __init__(
        self,
        coefficients: list[PlaintextTensor],
        degree: int,
        context: CryptoContext,
    ) -> None:
        ...
```

Evaluates a polynomial activation function of the given `degree` using the
baby-step/giant-step Paterson-Stockmeyer algorithm. Suitable for approximating
ReLU, sigmoid, GELU, or any smooth activation.

Two ready-made subclasses are provided for common use cases:

- **`SquareActivation`** — evaluates `f(x) = x²`. Degree 2, costs 1 level.
  Commonly used as a simple non-linearity in low-depth networks.
- **`ChebyshevReLU`** — evaluates a Chebyshev polynomial approximation of
  ReLU. Degree configurable; recommended degree 15 or 27 for high accuracy.

- `depth_cost()` returns `DepthCost.for_polynomial(degree)`.
- `required_rotations()` returns an empty `RotationSet`.

**Constructor parameters**

| Parameter      | Type                    | Description                                                               |
|----------------|-------------------------|---------------------------------------------------------------------------|
| `coefficients` | `list[PlaintextTensor]` | Encoded polynomial coefficients from `WeightEncoder.encode_polynomial_coefficients()`. |
| `degree`       | `int`                   | Degree of the polynomial. Determines depth cost.                          |
| `context`      | `CryptoContext`         | The owning context.                                                       |

```python
def forward(self, x: EncryptedTensor) -> EncryptedTensor:
    ...
```

---

### 6.9 `AdaptiveAvgPool2d`

```python
class AdaptiveAvgPool2d:
    def __init__(
        self,
        spatial_size: int,
        context: CryptoContext,
    ) -> None:
        ...
```

Computes the global average over the spatial dimensions of a feature map by
summing adjacent slots via `ceil(log2(spatial_size))` rotations, then
multiplying by `1 / spatial_size`. Mirrors `torch.nn.AdaptiveAvgPool2d` used
with an output size of `(1, 1)`.

- `depth_cost()` returns `DepthCost(levels=1)` (one rescaling for the final
  scalar multiply).
- `required_rotations()` returns `RotationSet.for_sum_reduction(spatial_size)`.

**Constructor parameters**

| Parameter      | Type            | Description                                            |
|----------------|-----------------|--------------------------------------------------------|
| `spatial_size` | `int`           | Total number of spatial elements to average over.      |
| `context`      | `CryptoContext` | The owning context.                                    |

```python
def forward(self, x: EncryptedTensor) -> EncryptedTensor:
    ...
```

---

### 6.10 `BatchNorm2d`

```python
class BatchNorm2d:
    def __init__(
        self,
        scale: PlaintextTensor,
        bias: PlaintextTensor,
        context: CryptoContext,
    ) -> None:
        ...
```

Applies batch normalisation as a fused scale-and-shift:
`output = scale * input + bias`. The `scale` and `bias` plaintexts encode the
combined effect of the learned gamma/beta parameters and the running
mean/variance statistics, folded together at export time. Mirrors
`torch.nn.BatchNorm2d`.

- `depth_cost()` returns `DepthCost(levels=1)` (one level for the
  ciphertext-plaintext multiply).
- `required_rotations()` returns an empty `RotationSet`.

**Constructor parameters**

| Parameter | Type              | Description                                          |
|-----------|-------------------|------------------------------------------------------|
| `scale`   | `PlaintextTensor` | Encoded per-channel scale factor.                    |
| `bias`    | `PlaintextTensor` | Encoded per-channel bias.                            |
| `context` | `CryptoContext`   | The owning context.                                  |

```python
def forward(self, x: EncryptedTensor) -> EncryptedTensor:
    ...
```

---

### 6.11 `Bootstrap`

```python
class Bootstrap:
    def __init__(self, context: CryptoContext) -> None:
        ...
```

Refreshes the multiplicative level budget of the input ciphertext using
approximate bootstrapping. Insert this module at any point in the network where
the accumulated depth cost would exceed `params.multiplicative_depth`.

`Bootstrap` requires that `KeyManager.generate_bootstrapping_keys()` was
called before inference.

- `depth_cost()` returns `DepthCost.for_bootstrap()` (0 — it restores levels
  rather than consuming them).
- `required_rotations()` returns a pre-determined set of rotation steps
  required by the bootstrapping circuit; these are added automatically when
  `generate_bootstrapping_keys()` is called.

```python
def forward(self, x: EncryptedTensor) -> EncryptedTensor:
    ...
```

---

## 7. Error Hierarchy

All SDK-level exceptions derive from `CKKSError`. Catch `CKKSError` to handle
any SDK error; catch specific subclasses for targeted recovery.

Raw exceptions from OpenFHE or FIDESlib are never allowed to propagate past the
SDK boundary. They are caught internally and re-raised as one of the types below.

```
CKKSError
├── LevelError
├── ShapeError
├── PackingError
├── KeyMissingError
└── ContextError
```

---

### `CKKSError`

```python
class CKKSError(Exception):
    ...
```

Base class. All SDK errors are instances of this class.

---

### `LevelError`

```python
class LevelError(CKKSError):
    def __init__(self, required: int, available: int) -> None:
        ...
```

Raised when an operation would consume more multiplicative levels than the
ciphertext has remaining.

| Attribute   | Type  | Description                                           |
|-------------|-------|-------------------------------------------------------|
| `required`  | `int` | Number of levels the attempted operation needs.       |
| `available` | `int` | Number of levels remaining in the ciphertext.         |

---

### `ShapeError`

```python
class ShapeError(CKKSError):
    def __init__(self, message: str) -> None:
        ...
```

Raised when tensor shapes are incompatible for the requested operation (e.g.,
adding two tensors with different `logical_shape`).

---

### `PackingError`

```python
class PackingError(CKKSError):
    def __init__(self, message: str) -> None:
        ...
```

Raised when a packing layout cannot accommodate the requested data, or when
the number of values to encrypt exceeds the available slot count.

---

### `KeyMissingError`

```python
class KeyMissingError(CKKSError):
    def __init__(self, missing_rotations: set[int]) -> None:
        ...
```

Raised when a rotation or evaluation key required by an operation was not
pre-generated.

| Attribute           | Type       | Description                                              |
|---------------------|------------|----------------------------------------------------------|
| `missing_rotations` | `set[int]` | The rotation steps for which no key exists.              |

---

### `ContextError`

```python
class ContextError(CKKSError):
    def __init__(self, message: str) -> None:
        ...
```

Raised when the `CryptoContext` is in an invalid or uninitialized state, for
example when `ContextBuilder.build()` is called without setting parameters.

---

## 8. Two-Phase Inference Pattern

FHE inference requires setup that would be impossible in plaintext ML:
rotation keys must be generated before any computation runs, and the total
multiplicative depth of the network must be known to set scheme parameters.
This creates a chicken-and-egg problem for a naïve graph executor.

FHE Engine resolves it by separating every module into two phases. The `Module`
protocol enforces this split: the first three methods (`depth_cost`,
`required_rotations`, `output_layout`) are pure metadata queries that require
no cryptographic state, while `forward` is the only method that touches
ciphertexts.

### Phase 1 — Static Analysis (no crypto)

```python
total_depth    = DepthCost.zero()
all_rotations  = RotationSet(steps=set())
current_layout = initial_input_layout

for module in network_modules:
    total_depth   = total_depth + module.depth_cost()
    all_rotations.merge(module.required_rotations(current_layout))
    current_layout = module.output_layout(current_layout)
```

After this loop, `total_depth.levels` is the value for
`CKKSParams.multiplicative_depth`, and `all_rotations` is passed directly to
`KeyManager.generate_rotation_keys()`. No ciphertext has been created yet.

When using `Sequential`, this entire loop is replaced by a single call:

```python
total_depth   = model.depth_cost()
all_rotations = model.required_rotations(initial_input_layout)
```

### Phase 2 — Inference (crypto)

```python
encrypted = context.encrypt(input_data, layout=initial_input_layout)

for module in network_modules:
    encrypted = module.forward(encrypted)

result = context.decrypt(encrypted)
```

The forward pass is a pure data flow: each module takes an `EncryptedTensor`
and returns one. The SDK handles packing strategy selection, rotation key
lookup, level tracking, and GPU dispatch internally. The caller never names a
rotation step, a modulus, or a level during inference.

### Putting It Together

```python
# -- Static analysis --
model = Sequential(
    Flatten(context),
    Linear(w1_diags, b1, context),
    PolyActivation(coeffs, degree=4, context=context),
    Linear(w2_diags, b2, context),
)

total_depth   = model.depth_cost()
all_rotations = model.required_rotations(input_layout)

params = CKKSParams(
    poly_modulus_degree=16384,
    multiplicative_depth=total_depth.levels,
    scaling_technique=ScalingTechnique.FLEXIBLEAUTO,
)
context = ContextBuilder().with_params(params).with_gpu().build()

key_manager = KeyManager(context)
key_manager.generate_key_pair()
key_manager.generate_relinearization_key()
key_manager.generate_rotation_keys(all_rotations)

# -- Inference --
encrypted_input  = context.encrypt(plaintext_data, layout=input_layout)
encrypted_output = model.forward(encrypted_input)
result           = context.decrypt(encrypted_output)
```
