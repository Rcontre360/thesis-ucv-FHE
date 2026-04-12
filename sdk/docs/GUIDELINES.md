# fhe_sdk Code Guidelines

Rules for all Python source files in `sdk/src/fhe_sdk/`.

---

## Imports

All imports must be at the top of the file — no inline or deferred imports inside functions or methods.

```python
# CORRECT
from fhe_sdk._backend import CKKSCiphertext, CKKSPlaintext
from fhe_sdk.plaintext import Plaintext

class Ciphertext:
    def __mul__(self, other):
        if isinstance(other, Plaintext): ...
```

```python
# WRONG — import inside method
def __mul__(self, other):
    from fhe_sdk.plaintext import Plaintext  # never do this
    ...
```

If a runtime circular import would result, use `TYPE_CHECKING` for type annotations only — not to defer actual logic:

```python
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from fhe_sdk.context import FHEContext  # only used in type hints
```

---

## Type dispatch

Use `isinstance` for type dispatch — never `hasattr`:

```python
# CORRECT
if isinstance(other, Plaintext): ...
elif isinstance(other, Ciphertext): ...

# WRONG
if hasattr(other, "_pt"): ...
```

---

## Class-level type annotations

All instance attributes must be declared with types at the class body level, before `__init__`. This mirrors the `FHEContext` style and makes the class interface self-documenting.

```python
class Plaintext:
    _context: "FHEContext"
    _pt: CKKSPlaintext
    _n_values: int

    def __init__(self, context, pt, n_values): ...
```

---

## Attribute naming

Private attributes should be named after what they hold, not after their implementation layer.

| Old (avoid) | Correct |
|---|---|
| `_backend_ct` | `_ct` |
| `_backend_pt` | `_pt` |
| `_original_size` | `_n_values` |
| `_backend_ctx` | `_backend_ctx` ← kept since it wraps a handle with no better name |

Do not encode the layer ("backend") into attribute names. Name attributes after their role.

---

## Method ordering

Within a class, order members as follows:

1. Class-level type annotations
2. `__init__`
3. Public methods (properties first, then regular, then dunder operators)
4. Private methods (`_` prefix) at the bottom

```python
class EncryptedVector:
    _context: "FHEContext"   # 1. annotations
    _ct: CKKSCiphertext

    def __init__(self, ...): ...   # 2. init

    @property
    def size(self) -> int: ...     # 3. public

    def decrypt(self) -> ...: ...
    def dot(self, ...) -> ...: ...
    def __add__(self, ...) -> ...: ...

    def _encode_and_align(self, ...): ...  # 4. private
    def _sum_slots(self, n: int): ...
```

---

## Comments

Only add a comment when the logic is not self-evident from the code. Do not use decorative section separators, and do not add docstrings to functions — the signature and surrounding code should be self-explanatory:

```python
# WRONG — visual noise, adds no information
# ------------------------------------------------------------------
# Arithmetic operators
# ------------------------------------------------------------------
def __add__(self, other): ...

# WRONG — docstring restating what the signature already says
def dot(self, weights: List[float]) -> "EncryptedVector":
    """Inner product with a plaintext weight vector.
    Returns an EncryptedVector of size 1 where slot[0] = sum(self[i] * weights[i]).
    """
    ...

# CORRECT — comment only where the why is non-obvious
def _sum_slots(self, n: int) -> "EncryptedVector":
    # rotation tree: each step doubles the number of slots accumulated
    step = 1
    while step < n:
        ...
```

---

## Mutating caller objects

Never silently mutate an object passed as an argument. If depth alignment is needed for a caller-provided `Plaintext`, raise a `ValueError` explaining what depth is required instead of calling `mod_drop_plain_inplace` on it.

```python
# CORRECT
if other._pt.depth != self._ct.depth:
    raise ValueError(f"Depth mismatch: ct depth={self._ct.depth}, pt depth={other._pt.depth}.")

# WRONG — mutates the caller's plaintext
while pt._pt.depth < ct.depth:
    ctx._ops.mod_drop_plain_inplace(pt._pt)
```

For list/scalar arguments (which we own after encoding), depth alignment via `mod_drop_plain_inplace` is fine since we hold the only reference.

---

## Class size

Each class covers one concept:

- `Plaintext` — an encoded (not encrypted) slot vector and its Python arithmetic
- `Ciphertext` — an encrypted slot vector and its homomorphic arithmetic
- `FHEContext` — the parameter set, key material, and encode/encrypt/decrypt API

Do not add methods to `FHEContext` that belong on `Ciphertext`, and vice versa.

---

## Not implemented yet

`nn.py` (layers, activations, sequential) is intentionally deferred. Do not add stubs or placeholders to the primitive files (`ciphertext.py`, `plaintext.py`, `context.py`). Remove any unfinished placeholder methods rather than leaving them with `# TODO` or partial implementations.

---

## Tests

Every public method must have at least one test in `sdk/tests/`. Tests are organized by file:

- `test_context.py` — `FHEContext` builder, encode, encrypt, decrypt
- `test_plaintext.py` — `Plaintext` arithmetic
- `test_ciphertext.py` — `Ciphertext` arithmetic, depth mismatch errors

Tests that require the compiled `_backend` use the `built_context` fixture from `conftest.py`. Tests that only exercise pure Python logic (parameter validation, error messages) run without the fixture and do not require GPU.

Run with:
```bash
cd sdk
pytest
```
