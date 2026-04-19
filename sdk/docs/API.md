# fhe_sdk API Reference

## Table of Contents

1. [Enums](#enums)
2. [FHEContext](#fhecontext)
3. [Plaintext](#plaintext)
4. [Ciphertext](#ciphertext)
5. [fhe_sdk.nn](#fhe_sdknn)
   - [Module](#module-base-class)
   - [Linear](#linear)
   - [Square](#square)
   - [ApproxReLU](#approxrelu)
   - [ApproxSigmoid](#approxsigmoid)
   - [Sequential](#sequential)

---

## Enums

```python
import enum

class SecurityLevel(enum.IntEnum):
    SEC128 = ...   # 128-bit post-quantum security (default)
    SEC192 = ...   # 192-bit post-quantum security
    SEC256 = ...   # 256-bit post-quantum security

class KeyswitchingType(enum.IntEnum):
    METHOD_I  = ...  # Standard key-switching (lower noise, default)
    METHOD_II = ...  # Hybrid key-switching (faster, slightly higher noise)
```

`NONE` is not exposed. Every context requires an explicit security level.

---

## FHEContext

`fhe_sdk.context.FHEContext`

Top-level object. Owns all cryptographic state (keys, encoder, encryptor, decryptor). Created once per session. All `Plaintext` and `Ciphertext` objects are bound to the context that produced them.

```python
class FHEContext:
    def __init__(self) -> None: ...
```

### Configuration setters

All setters return `self` for chaining. They raise `RuntimeError` if called after `build()`.

| Method | Parameter | Type | Description |
|---|---|---|---|
| `set_poly_modulus_degree` | `degree` | `int` | Polynomial ring degree. Must be a power of 2. Common values: `4096`, `8192`, `16384`, `32768`. |
| `set_coeff_modulus_bit_sizes` | `bit_sizes` | `list[int]` | Bit sizes of each prime in the coefficient modulus chain. First and last entries are typically 60 bits; interior primes should match the scale. |
| `set_scale` | `scale` | `float` | Encoding scale. Common values: `2**40`, `2**50`. Must match the interior primes of `coeff_modulus_bit_sizes`. |
| `set_security_level` | `level` | `SecurityLevel` | Security parameter set. Defaults to `SecurityLevel.SEC128`. |
| `set_keyswitching_type` | `ktype` | `KeyswitchingType` | Key-switching algorithm. Defaults to `KeyswitchingType.METHOD_I`. |

```python
def set_poly_modulus_degree(self, degree: int) -> "FHEContext": ...
def set_coeff_modulus_bit_sizes(self, bit_sizes: list[int]) -> "FHEContext": ...
def set_scale(self, scale: float) -> "FHEContext": ...
def set_security_level(self, level: SecurityLevel) -> "FHEContext": ...
def set_keyswitching_type(self, ktype: KeyswitchingType) -> "FHEContext": ...
```

### `build()`

```python
def build(self) -> "FHEContext": ...
```

Validates all parameters, generates the CKKS context, and produces the secret key, public key, and relinearization key. Must be called before `encode`, `encrypt`, or `decrypt`. Returns `self`.

Raises `ValueError` if any required parameter (`poly_modulus_degree`, `coeff_modulus_bit_sizes`, `scale`) has not been set.

After `build()` is called, all setter methods raise:

```
RuntimeError: Context already built — create a new FHEContext to change parameters.
```

### `FHEContext.default()`

```python
@classmethod
def default(cls) -> "FHEContext": ...
```

Convenience factory. Equivalent to:

```python
(
    FHEContext()
    .set_poly_modulus_degree(8192)
    .set_coeff_modulus_bit_sizes([60, 40, 40, 60])
    .set_scale(2**40)
    .set_security_level(SecurityLevel.SEC128)
    .set_keyswitching_type(KeyswitchingType.METHOD_I)
    .build()
)
```

Provides 2 usable multiplication levels, 128-bit security, and up to 4096 slots.

### `encode()`

```python
def encode(self, values: list[float]) -> "Plaintext": ...
```

Encodes `values` using CKKS slot encoding at the configured scale. Returns a `Plaintext` bound to this context. The result is **not encrypted**.

| Parameter | Type | Description |
|---|---|---|
| `values` | `list[float]` | Values to encode. Length must not exceed `poly_modulus_degree // 2`. |

### `encrypt()`

```python
def encrypt(self, values: "list[float] | Plaintext") -> "Ciphertext": ...
```

Encrypts under the public key. Returns a `Ciphertext` bound to this context.

- `list[float]`: encodes then encrypts (shorthand for `ctx.encrypt(ctx.encode(values))`).
- `Plaintext`: encrypts the already-encoded plaintext directly.

| Parameter | Type | Description |
|---|---|---|
| `values` | `list[float] \| Plaintext` | Values or encoded plaintext to encrypt. If `list[float]`, length must not exceed `poly_modulus_degree // 2`. |

### `decrypt()`

```python
def decrypt(self, ciphertext: "Ciphertext") -> list[float]: ...
```

Decrypts and decodes. Returns a `list[float]` of the same length as the original plaintext.

| Parameter | Type | Description |
|---|---|---|
| `ciphertext` | `Ciphertext` | Must have been produced by this context. |

Raises `ValueError` if `ciphertext` was produced by a different `FHEContext`.

### `decode()`

```python
def decode(self, plaintext: "Plaintext") -> list[float]: ...
```

Decodes a `Plaintext` back to `list[float]`.

| Parameter | Type | Description |
|---|---|---|
| `plaintext` | `Plaintext` | Must have been produced by this context. |

---

## Plaintext

`fhe_sdk.plaintext.Plaintext`

An encoded (not encrypted) vector. Represents values transformed into the CKKS polynomial ring at a fixed scale. Never constructed directly — always obtained from `ctx.encode()` or `Plaintext` arithmetic.

```python
class Plaintext:
    # Not user-constructible.
    ...
```

### Arithmetic operators

All operations stay in the plaintext domain. Each returns a new `Plaintext`.

```python
# Plaintext OP Plaintext
def __add__(self, other: "Plaintext") -> "Plaintext": ...
def __sub__(self, other: "Plaintext") -> "Plaintext": ...
def __mul__(self, other: "Plaintext") -> "Plaintext": ...

# Plaintext OP list[float] | float
def __add__(self, other: "list[float] | float") -> "Plaintext": ...
def __sub__(self, other: "list[float] | float") -> "Plaintext": ...
def __mul__(self, other: "list[float] | float") -> "Plaintext": ...

# Reflected (list[float] | float OP Plaintext)
def __radd__(self, other: "list[float] | float") -> "Plaintext": ...
def __rsub__(self, other: "list[float] | float") -> "Plaintext": ...
def __rmul__(self, other: "list[float] | float") -> "Plaintext": ...
```

### `decode()`

```python
def decode(self) -> list[float]: ...
```

Shorthand for `context.decode(self)`.

### `size`

```python
@property
def size(self) -> int: ...
```

Number of slots (equal to the length of the original `values` passed to `ctx.encode()`).

---

## Ciphertext

`fhe_sdk.ciphertext.Ciphertext`

An encrypted vector. Never instantiated directly — always obtained from `ctx.encrypt()` or arithmetic operations.

```python
class Ciphertext:
    # Not user-constructible.
    ...
```

### Arithmetic operators

Each operator returns a new `Ciphertext`. Operands are not modified.

```python
# Ciphertext OP Ciphertext
def __add__(self, other: "Ciphertext") -> "Ciphertext": ...   # homomorphic addition
def __sub__(self, other: "Ciphertext") -> "Ciphertext": ...   # homomorphic subtraction
def __mul__(self, other: "Ciphertext") -> "Ciphertext": ...   # element-wise homomorphic multiply

# Ciphertext OP Plaintext | list[float] | float
# list[float] and float are auto-encoded at the context's scale before the operation.
def __add__(self, other: "Plaintext | list[float] | float") -> "Ciphertext": ...
def __sub__(self, other: "Plaintext | list[float] | float") -> "Ciphertext": ...
def __mul__(self, other: "Plaintext | list[float] | float") -> "Ciphertext": ...

# Reflected (Plaintext | list[float] | float OP Ciphertext)
def __radd__(self, other: "Plaintext | list[float] | float") -> "Ciphertext": ...
def __rsub__(self, other: "Plaintext | list[float] | float") -> "Ciphertext": ...
def __rmul__(self, other: "Plaintext | list[float] | float") -> "Ciphertext": ...
```

### `__rmatmul__`

```python
def __rmatmul__(self, W: "list[list[float]]") -> "Ciphertext": ...
```

Computes `W @ ct` where `W` is a plaintext matrix of shape `(out, in)`. Returns a `Ciphertext` of size `out`. Invoked via `W @ ct`. Uses the diagonal encoding method internally.

### `dot()`

```python
def dot(self, other: "Ciphertext | Plaintext | list[float]") -> "Ciphertext": ...
```

Inner product. Returns a single-slot `Ciphertext` whose first slot holds the result. `Plaintext` or `list[float]` operands use the cheaper ciphertext-plaintext path.

### Operation depth table

| Operation | Modulus levels consumed | Notes |
|---|---|---|
| `ct + ct` | 0 | Homomorphic addition |
| `ct + Plaintext` | 0 | Plaintext addition |
| `ct + list[float]` | 0 | Auto-encodes then adds |
| `ct - ct` | 0 | Homomorphic subtraction |
| `ct - Plaintext` | 0 | Plaintext subtraction |
| `ct * Plaintext` | 1 | Plaintext multiply + rescale |
| `ct * list[float]` | 1 | Auto-encodes then multiplies |
| `ct * ct` | 1 | Homomorphic multiply + auto relinearize + rescale |
| `W @ ct` | 1 | Plaintext matrix; same cost as `ct * Plaintext` |
| `ct.dot(Plaintext)` | 1 | Plaintext dot + rescale |
| `ct.dot(ct)` | 1 | Multiply + relinearize + rescale + rotation sum |

Relinearization and rescaling are automatic. Users never call them explicitly.

### `decrypt()`

```python
def decrypt(self) -> list[float]: ...
```

Shorthand for `context.decrypt(self)`.

### `size`

```python
@property
def size(self) -> int: ...
```

Number of plaintext slots (equal to the length of the original `values` passed to `ctx.encrypt()`).

---

## fhe_sdk.nn

PyTorch-inspired neural network primitives. All layers take a `Ciphertext` as input and return a `Ciphertext`.

```python
from fhe_sdk.nn import Sequential, Linear, Square, ApproxReLU, ApproxSigmoid
```

---

### Module (base class)

`fhe_sdk.nn.Module`

```python
class Module:
    def forward(self, x: Ciphertext) -> Ciphertext: ...
    def __call__(self, x: Ciphertext) -> Ciphertext: ...  # delegates to forward
```

All layers inherit from `Module` and override `forward`.

---

### Linear

`fhe_sdk.nn.linear.Linear`

Fully-connected layer. Computes `W @ x + b` where `W` and `b` are plaintext.

```python
class Linear(Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True) -> None: ...

    def load_weights(
        self,
        weight: ArrayLike,
        bias: ArrayLike | None = None,
    ) -> None: ...

    def load_from_torch(self, layer: "torch.nn.Linear") -> None: ...

    def forward(self, x: Ciphertext) -> Ciphertext: ...
```

**Constructor parameters:**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `in_features` | `int` | — | Expected number of slots in the input `Ciphertext`. |
| `out_features` | `int` | — | Number of slots in the output `Ciphertext`. |
| `bias` | `bool` | `True` | If `False`, `load_weights` ignores any bias argument and `forward` omits the bias addition. |

**`load_weights` parameters:**

| Parameter | Type | Description |
|---|---|---|
| `weight` | `ArrayLike` | Weight matrix of shape `(out_features, in_features)`. |
| `bias` | `ArrayLike \| None` | Bias vector of shape `(out_features,)`. Required when `bias=True`. |

`ArrayLike`: `numpy.ndarray`, `torch.Tensor`, `list[list[float]]`, or any object with a `.tolist()` method. `.tolist()` is called automatically when present.

Raises `ValueError` if weight shape does not match `(out_features, in_features)`.

**`load_from_torch`:**

```python
def load_from_torch(self, layer: "torch.nn.Linear") -> None: ...
```

Sugar for `self.load_weights(layer.weight.detach(), layer.bias.detach())`. Raises `ImportError` if `torch` is not installed.

**`forward`:** Computes `W @ x + b`. Bias addition is a free plaintext operation (no depth cost). Raises `RuntimeError` if weights have not been loaded.

---

### Square

`fhe_sdk.nn.activation.Square`

Applies `f(x) = x²` element-wise. Exact. Consumes 1 modulus level.

```python
class Square(Module):
    def forward(self, x: Ciphertext) -> Ciphertext: ...
```

No constructor parameters.

---

### ApproxReLU

`fhe_sdk.nn.activation.ApproxReLU`

Minimax polynomial approximation of ReLU over `[-bound, bound]`.

```python
class ApproxReLU(Module):
    def __init__(self, degree: int = 3, bound: float = 5.0) -> None: ...
    def forward(self, x: Ciphertext) -> Ciphertext: ...
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `degree` | `int` | `3` | Polynomial degree. Must be odd and ≥ 3. Higher degree = better approximation, more depth. |
| `bound` | `float` | `5.0` | Approximation interval `[-bound, bound]`. Values outside this range produce undefined results. |

---

### ApproxSigmoid

`fhe_sdk.nn.activation.ApproxSigmoid`

Minimax polynomial approximation of `σ(x) = 1 / (1 + e^−x)` over `[-bound, bound]`.

```python
class ApproxSigmoid(Module):
    def __init__(self, degree: int = 3, bound: float = 5.0) -> None: ...
    def forward(self, x: Ciphertext) -> Ciphertext: ...
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `degree` | `int` | `3` | Polynomial degree. Must be odd and ≥ 3. |
| `bound` | `float` | `5.0` | Approximation interval `[-bound, bound]`. |

**Activation depth comparison:**

| Activation | Levels consumed | Notes |
|---|---|---|
| `Square` | 1 | Exact. Most depth-efficient. |
| `ApproxReLU(degree=3)` | 2 | Baby-step Giant-step evaluation. |
| `ApproxReLU(degree=5)` | 3 | Better ReLU approximation. |
| `ApproxSigmoid(degree=3)` | 2 | Same cost as `ApproxReLU(degree=3)`. |
| `ApproxSigmoid(degree=5)` | 3 | Same cost as `ApproxReLU(degree=5)`. |

A degree-`d` polynomial requires `ceil(log2(d))` levels with Baby-step Giant-step (used internally), vs. `d - 1` levels naively.

---

### Sequential

`fhe_sdk.nn.sequential.Sequential`

Ordered container of `Module` layers. Applies each in insertion order.

```python
class Sequential(Module):
    def __init__(self, *layers: Module) -> None: ...
    def forward(self, x: Ciphertext) -> Ciphertext: ...
    def __getitem__(self, index: int) -> Module: ...
    def __len__(self) -> int: ...
```

`forward` passes `x` through each layer in order; the output of layer `i` is the input to layer `i+1`. Raises `IndexError` for out-of-range index access.
