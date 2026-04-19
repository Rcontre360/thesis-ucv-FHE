# fhe-sdk Code Guidelines

Rules for all Python source files in `sdk/src/`.

---

## Imports

All imports must be at the top of the file — no inline or deferred imports inside functions or methods.

```python
# CORRECT
from core._backend import CKKSCiphertext, CKKSPlaintext
from core.plaintext import PlaintextVector

class EncryptedVector:
    def __mul__(self, other):
        if isinstance(other, PlaintextVector): ...
```

```python
# WRONG — import inside method
def __mul__(self, other):
    from core.plaintext import PlaintextVector  # never do this
    ...
```

If a runtime circular import would result, use `TYPE_CHECKING` for type annotations only — not to defer actual logic:

```python
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from api.context import FHEContext  # only used in type hints
```

---

## Type dispatch

Use `isinstance` for type dispatch — never `hasattr`:

```python
# CORRECT
if isinstance(other, PlaintextVector): ...
elif isinstance(other, EncryptedVector): ...

# WRONG
if hasattr(other, "_pt"): ...
```

---

## Class-level type annotations

All instance attributes must be declared with types at the class body level, before `__init__`. This mirrors the `FHEContext` style and makes the class interface self-documenting.

```python
class EncryptedVector:
    _context: "FHEContext"
    _ct: CKKSCiphertext
    _n_values: int

    def __init__(self, context, ct, n_values): ...
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
    """Inner product with a plaintext weight vector."""
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

Never silently mutate an object passed as an argument. If depth alignment is needed for a caller-provided `PlaintextVector`, raise a `ValueError` explaining what depth is required instead of calling `mod_drop_plain_inplace` on it.

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

- `PlaintextVector` — an encoded (not encrypted) slot vector and its Python arithmetic
- `EncryptedVector` — an encrypted slot vector and its homomorphic arithmetic
- `FHEContext` — the parameter set, key material, and encode/encrypt/decrypt API

Do not add methods to `FHEContext` that belong on `EncryptedVector`, and vice versa.

---

## Tests

Every public method must have at least one test. Each module gets its own dedicated test file — never mix tests for different modules in the same file.

```
tests/
  test_context.py     # FHEContext only
  test_ciphertext.py  # EncryptedVector only
  test_plaintext.py   # PlaintextVector only
  test_tensor.py      # PlaintextTensor only
  test_relu.py        # api/functions/activations — ReLU only
  test_linear.py      # api/layers/linear — Linear only
  test_sequential.py  # api/sequential — Sequential only
```

Tests assume the SDK is already installed in the active Python environment. They import directly from the installed packages — no `sys.path` manipulation, no relative imports from `src/`.

Tests that require the compiled backend use the `built_context` fixture from `conftest.py`. Tests that only exercise pure Python logic (parameter validation, error messages) run without the fixture and do not require GPU.

Install and run:
```bash
bash scripts/run_tests.sh   # installs the SDK then runs pytest
```

---

## Build process

All build artifacts — compiled binaries, CMake cache, installed third-party libraries — must stay inside the `build/` folder at the SDK root. Nothing is written outside that directory during a build.

```
sdk/
  build/          ← all build output lives here
    heongpu/      ← HEonGPU installed libraries
    src/backend/  ← compiled _backend.so
  src/            ← source only, never modified by the build
  external/       ← submodules only, never modified by the build
```

CMake and the backend build scripts (`build.sh`, `install_system_deps.sh`) are strictly for compiling C++/CUDA code and external dependencies. They must never install or manipulate pure Python packages — that is the responsibility of `pip` and `pyproject.toml`.

---

## Packaging

Python package distribution is handled entirely through `pyproject.toml` using `scikit-build-core`. Pure Python packages (`api/`, `core/`) are declared via `wheel.packages`; the compiled extension (`_backend.so`) is installed via a CMake `install()` rule with component `python_modules`.

Do not duplicate packaging logic in shell scripts (e.g. manual `cp` of `.py` files to site-packages, or `cmake --install` as a substitute for `pip install`).

---

## Examples

Examples live in `examples/` and assume the SDK is already installed in the user's active Python environment. They must import directly from the installed packages with no path manipulation:

```python
# CORRECT
from api import FHEContext
from api.layers.linear import Linear

# WRONG — never manipulate sys.path or assume a specific venv location
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
```

Run an example with:
```bash
bash scripts/run_example.sh <example_name>
```

The script sets `LD_LIBRARY_PATH` for the HEonGPU shared libraries and respects the `PYTHON` environment variable to target any Python installation — never hardcodes a path like `.env/bin/python`.
