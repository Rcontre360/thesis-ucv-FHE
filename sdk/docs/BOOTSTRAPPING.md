# fhe_sdk Bootstrapping Design Specification

This document is the authoritative design spec and developer guide for bootstrapping
integration in `fhe_sdk`. It covers how HEonGPU's bootstrapping backend works, how the
Python SDK wraps it, and the concrete API design for automatic bootstrapping in
`FHEContext` and `Sequential`.

Audience: developers implementing `context.py`, `ciphertext.py`, and
`nn/sequential.py` who need to understand both the HEonGPU internals and the desired
Python API behavior. For a general introduction to the cryptographic parameters and
the level/depth model, see [API.md](API.md) and the Depth Budget Guide in
[README.md](../README.md).

---

## Table of Contents

1. [Overview](#1-overview)
2. [HEonGPU Bootstrapping API (Backend Layer)](#2-heongpu-bootstrapping-api-backend-layer)
3. [SDK-Level Design: Transparent Bootstrapping](#3-sdk-level-design-transparent-bootstrapping)
4. [Parameter Sizing with Bootstrapping](#4-parameter-sizing-with-bootstrapping)
5. [FHEContext.build() Changes Needed](#5-fhecontextbuild-changes-needed)
6. [User-Facing ctx.bootstrap() Method](#6-user-facing-ctxbootstrap-method)
7. [Limitations and Known Constraints](#7-limitations-and-known-constraints)

---

## 1. Overview

CKKS ciphertexts are defined relative to a chain of modulus primes
`q_0 * q_1 * ... * q_L`. Every multiplication (or plaintext multiply with rescale)
consumes one prime. When the last interior prime is consumed, the ciphertext is at
level 0 and no further multiplications are possible. For a network with `k` activation
layers that each cost `d` levels, you need at least `k * d` interior primes in the
chain, which quickly requires a large `poly_modulus_degree` for security reasons.

Bootstrapping is the procedure that "refreshes" a level-0 ciphertext back to a
ciphertext near the top of the modulus chain, enabling further computation. It does
this by evaluating a polynomial circuit homomorphically that corrects for the modular
reduction error introduced when the modulus is raised. The mathematical pipeline is:
ModRaise → CoeffToSlot (homomorphic DFT) → EvalMod (approximate sine) → SlotToCoeff
(homomorphic inverse DFT). HEonGPU executes this entire circuit on the GPU, taking
roughly 100-170 ms for N=2^16 on an RTX 4090.

The tradeoff is clear: without bootstrapping, each activation in the network
permanently consumes levels, and the chain must be long enough to cover the entire
network. With bootstrapping, the chain only needs to be long enough to cover the
bootstrapping circuit itself (5-10 levels) plus the levels needed between bootstraps.
This makes deep network inference feasible at smaller `poly_modulus_degree` values, at
the cost of periodic ~100 ms pauses during inference. For most GPU inference workloads
where the alternative is a much larger N, the tradeoff is favorable.

The two bootstrapping variants exposed by the SDK are:

- `BootstrappingType.REGULAR` — standard CoeffToSlot/EvalMod/SlotToCoeff pipeline,
  supports full complex packing.
- `BootstrappingType.SLIM` — starts with SlotToCoeff before ModRaise, consuming fewer
  levels overall; appropriate for real-valued networks.

A third backend variant, `regular_bootstrapping_v2`, is available for non-sparse key
distributions but has additional key management requirements; it is exposed at the raw
`_backend` level only and is not part of the high-level SDK bootstrapping path.

---

## 2. HEonGPU Bootstrapping API (Backend Layer)

This section describes the raw `_backend` bindings for bootstrapping. SDK developers
implementing `context.py` must understand this layer completely. Users of the SDK never
call these directly.

### 2.1 Setup Flow (One-Time, Before Inference)

Bootstrapping requires a three-step setup that must complete before the first call to
`regular_bootstrapping()`. This setup is expensive but happens only once per context.

**Step 1 — Precompute the bootstrapping matrices:**

```python
from fhe_sdk._backend import (
    CKKSContext, CKKSOperator, CKKSEncoder,
    BootstrappingConfig, BootstrappingType,
)

config = BootstrappingConfig(
    CtoS_piece=3,      # CoeffToSlot DFT baby-step splits (range 2-5)
    StoC_piece=3,      # SlotToCoeff DFT baby-step splits (range 2-5)
    taylor_number=11,  # EvalMod sine Taylor series terms (range 6-15)
    less_key_mode=False,
)

# operator is a CKKSOperator already bound to the context and encoder.
# scale is the same 2**40 (or 2**50) used everywhere in the computation.
operator.generate_bootstrapping_params(
    scale=2**40,
    config=config,
    boot_type=BootstrappingType.REGULAR,
)
```

This call precomputes and stores on the GPU the CoeffToSlot Vandermonde matrices,
SlotToCoeff matrices, and the EvalMod polynomial coefficients, decomposed for
Baby-Step Giant-Step (BSGS) rotation-efficient evaluation. It does not generate any
keys. It must be called before `bootstrapping_key_indexs()`.

**Step 2 — Retrieve the required Galois shift indices:**

```python
boot_shifts: list[int] = operator.bootstrapping_key_indexs()
# Example result: [1, 2, 4, 8, 16, ..., 512, -1, -2, -4, ...]
```

The bootstrapping circuit applies specific cyclic rotations during the DFT stages. The
exact set of shifts depends on `CtoS_piece`, `StoC_piece`, and `poly_modulus_degree`.
`bootstrapping_key_indexs()` returns the complete list. Calling it before
`generate_bootstrapping_params()` raises a C++ exception that propagates as a
`RuntimeError` in Python.

The returned shift list must be merged with any shifts required by the network's
`Linear` layers (diagonal-method rotations). The combined list is used to generate a
single `CKKSGaloiskey`.

**Step 3 — Generate a Galois key covering all required shifts:**

```python
from fhe_sdk._backend import CKKSGaloiskey, CKKSKeyGenerator

# network_shifts: shifts needed by the Linear layers (from diagonal method)
all_shifts = list(set(boot_shifts + network_shifts))

gk = CKKSGaloiskey(context, all_shifts)  # exact-shift constructor
keygen = CKKSKeyGenerator(context)
keygen.generate_galois_key(gk, sk)       # fills gk from the secret key
```

The three-argument constructor `CKKSGaloiskey(context, shifts)` creates a key that
covers exactly the rotation amounts in `shifts`. Generating keys for shifts not in this
list later is not possible without rebuilding the key; this is why the merge must
happen before key generation.

### 2.2 Runtime Bootstrapping

Once setup is complete, refreshing a ciphertext that has reached level 0 is a single
call:

```python
# ct_exhausted is at level 0 (no more multiply levels available)
ct_refreshed = operator.regular_bootstrapping(ct_exhausted, gk, rk)
```

`regular_bootstrapping` returns a new `CKKSCiphertext` at near-full depth. The input
`ct_exhausted` is not modified. The refreshed ciphertext has slightly more noise than
the original fresh ciphertext due to the approximation errors accumulated by the
EvalMod polynomial. This noise increase is bounded and acceptable for inference
workloads, but it is not zero.

The slim variant is called identically:

```python
ct_refreshed = operator.slim_bootstrapping(ct_exhausted, gk, rk)
```

Slim bootstrapping uses the pipeline SlotToCoeff → ModRaise → CoeffToSlot → EvalMod,
which reduces the level budget consumed by the bootstrapping circuit itself. For
real-valued networks this is the preferred variant.

### 2.3 Level Budget Constraint

A critical and easy-to-overlook constraint: the context's modulus chain must be long
enough to contain both the network levels and the levels consumed by the bootstrapping
circuit itself. The bootstrapping circuit consumes levels during CoeffToSlot, EvalMod,
and SlotToCoeff. With default `BootstrappingConfig` parameters (piece=3, taylor=11),
the circuit consumes approximately 8-10 levels.

The level budget after bootstrapping is:

```
levels_available_after_boot = total_usable_levels - levels_consumed_by_boot_circuit
```

If the bootstrapping circuit needs 10 levels and the chain has only 6 usable levels,
`generate_bootstrapping_params()` will fail or produce a non-functional setup. The
chain must be sized for the bootstrapping circuit first, and then evaluated for how
many network levels remain.

Section 4 of this document provides a concrete sizing table.

---

## 3. SDK-Level Design: Transparent Bootstrapping

The goal is to hide all of Section 2 from the end user. A user with a deep network
should be able to write:

```python
from fhe_sdk import FHEContext
from fhe_sdk.nn import Sequential, Linear, ApproxReLU

ctx = (
    FHEContext()
    .set_poly_modulus_degree(16384)
    .set_coeff_modulus_bit_sizes([60, 40, 40, 40, 40, 40, 40, 40, 60])
    .set_scale(2**40)
    .enable_bootstrapping()
    .build()
)

model = Sequential(
    Linear(128, 64), ApproxReLU(degree=3),
    Linear(64, 64),  ApproxReLU(degree=3),
    Linear(64, 64),  ApproxReLU(degree=3),
    Linear(64, 10),
)

enc_input = ctx.encrypt(plaintext_input)
enc_output = model(enc_input)   # bootstrapping happens automatically mid-network
result = enc_output.decrypt()
```

The following subsections specify every design decision needed to make this work.

### 3.1 FHEContext.enable_bootstrapping()

A new setter method on `FHEContext`, following the existing fluent-setter pattern:

```python
def enable_bootstrapping(
    self,
    config: BootstrappingConfig | None = None,
    boot_type: BootstrappingType = BootstrappingType.REGULAR,
    policy: BootstrappingPolicy = BootstrappingPolicy.AUTO,
) -> "FHEContext": ...
```

Calling `enable_bootstrapping()` before `build()` stores three things on the context
instance:

- `self._boot_config`: the `BootstrappingConfig` to use (defaults to
  `BootstrappingConfig()` with piece=3, taylor=11, less_key_mode=False).
- `self._boot_type`: `BootstrappingType.REGULAR` or `BootstrappingType.SLIM`.
- `self._boot_policy`: a `BootstrappingPolicy` value (see Section 3.2).
- `self._bootstrapping_enabled = True`.

It does not call any backend code. All backend setup happens in `build()`. Like all
setters it raises `RuntimeError` if called after `build()` and returns `self`.

### 3.2 BootstrappingPolicy

A new enum in `fhe_sdk.enums`:

```python
import enum

class BootstrappingPolicy(enum.Enum):
    AUTO = "auto"
    MANUAL = "manual"
    THRESHOLD = "threshold"
```

The policy controls when `Sequential.forward()` decides to bootstrap between layers:

- `AUTO` — bootstrap whenever `ct.level <= 1`. This is the safe default: it leaves
  one level of headroom above 0 to avoid accidentally calling a multiply on a level-0
  ciphertext before the bootstrap can occur.

- `MANUAL` — `Sequential.forward()` never bootstraps. The user is responsible for
  calling `ctx.bootstrap(ct)` at the right point, either between layers in a custom
  `Module.forward()` or outside the model. Useful when the user has an unusual circuit
  that doesn't fit neatly into the sequential pattern.

- `THRESHOLD(n)` — bootstrap whenever `ct.level <= n`. Useful when a layer known to
  consume multiple levels is about to run and the user wants to ensure at least `n`
  levels of margin before it.

`THRESHOLD` requires a level value at construction time. A convenience constructor is
needed:

```python
@staticmethod
def threshold(n: int) -> "BootstrappingPolicy": ...
```

Or, more concisely, `THRESHOLD` can be a class that takes `n` in its constructor.
The simplest Pythonic implementation uses a dataclass sentinel:

```python
from dataclasses import dataclass

class BootstrappingPolicy(enum.Enum):
    AUTO = "auto"
    MANUAL = "manual"

@dataclass(frozen=True)
class BootstrappingThreshold:
    """Bootstrap when ct.level <= min_level."""
    min_level: int
```

`Sequential` checks for `isinstance(policy, BootstrappingThreshold)` to handle the
threshold case. This avoids the awkwardness of an enum member that carries data.

Usage:

```python
ctx.enable_bootstrapping(policy=BootstrappingPolicy.AUTO)
ctx.enable_bootstrapping(policy=BootstrappingThreshold(min_level=2))
ctx.enable_bootstrapping(policy=BootstrappingPolicy.MANUAL)
```

### 3.3 Ciphertext.level Property

The `_backend.CKKSCiphertext` already exposes a `.level` property (see
`bind_data.cu`, line 67):

```
level: remaining usable multiplication levels = coeff_modulus_count - (depth + 1)
```

The Python `Ciphertext` wrapper must expose this as a read-only property that
delegates to the underlying `CKKSCiphertext`:

```python
@property
def level(self) -> int:
    """Remaining usable multiplication levels in this ciphertext."""
    return self._raw.level
```

A fresh ciphertext encrypted by a context with 8 usable levels has `level == 8`. After
two `ApproxReLU(degree=3)` activations (4 levels consumed) it has `level == 4`. At
`level == 0` no further multiplications are possible.

### 3.4 Sequential.forward() with Automatic Bootstrapping

This is the core integration point. The current `Sequential.forward()` is:

```python
def forward(self, x: Ciphertext) -> Ciphertext:
    for layer in self._layers:
        x = layer(x)
    return x
```

The new version checks the ciphertext level after each layer and bootstraps if the
policy says to. The bootstrapping decision belongs in `Sequential`, not in individual
`Module` subclasses, because `Sequential` has the context reference and the policy.
Individual modules should be context-unaware.

The recommended design is Option A from the problem statement: `Sequential.forward()`
checks `.level` after each layer and calls `ctx.bootstrap(ct)` when the policy
triggers. This is explicit, visible in tracebacks, and easy to test.

Option B (a context backref on `Ciphertext` that auto-bootstraps on the next multiply)
is rejected because it makes the bootstrapping point invisible to the caller, creates
a circular reference between `Ciphertext` and `FHEContext`, and makes it impossible to
distinguish intentional level-0 operations from accidental ones.

The new `Sequential.__init__` must accept an optional context:

```python
class Sequential(Module):
    def __init__(self, *layers: Module, ctx: "FHEContext | None" = None) -> None:
        self._layers = list(layers)
        self._ctx = ctx
```

And `forward()` becomes:

```python
def forward(self, x: Ciphertext) -> Ciphertext:
    for layer in self._layers:
        x = layer(x)
        if self._ctx is not None and self._ctx.bootstrapping_enabled:
            x = self._maybe_bootstrap(x)
    return x

def _maybe_bootstrap(self, x: Ciphertext) -> Ciphertext:
    policy = self._ctx.boot_policy
    if isinstance(policy, BootstrappingThreshold):
        if x.level <= policy.min_level:
            return self._ctx.bootstrap(x)
    elif policy is BootstrappingPolicy.AUTO:
        if x.level <= 1:
            return self._ctx.bootstrap(x)
    # BootstrappingPolicy.MANUAL: never bootstrap here
    return x
```

The check fires after every layer. When the policy does not trigger (level is healthy),
`_maybe_bootstrap` is a near-zero-cost comparison. When it does trigger, it calls
`ctx.bootstrap(ct)`, which is the public method specified in Section 6.

If `ctx` is `None` or bootstrapping is not enabled, `Sequential.forward()` behaves
exactly as before.

### 3.5 Registering the Context with Sequential

The user must either pass `ctx` to `Sequential` at construction time, or call a method
after construction. Two patterns are supported:

**Construction-time:**

```python
model = Sequential(
    Linear(128, 64), ApproxReLU(degree=3),
    Linear(64, 64),  ApproxReLU(degree=3),
    ctx=ctx,
)
```

**Post-construction (useful when building the model before building the context):**

```python
model = Sequential(
    Linear(128, 64), ApproxReLU(degree=3),
    Linear(64, 64),  ApproxReLU(degree=3),
)
model.set_context(ctx)
```

`set_context` is a method on `Sequential` that sets `self._ctx`. It can be called
multiple times (replacing the context), but raises `RuntimeError` if called during an
active forward pass (i.e., inside a recursive `forward()` call — this edge case is
unlikely but worth guarding).

### 3.6 Worked Example: Full Pipeline

```python
from fhe_sdk import FHEContext
from fhe_sdk.nn import Sequential, Linear, ApproxReLU
from fhe_sdk._backend import BootstrappingConfig, BootstrappingType

# 1. Build context with bootstrapping enabled.
#    The coeff modulus chain must be long enough — see Section 4 for sizing.
ctx = (
    FHEContext()
    .set_poly_modulus_degree(16384)
    .set_coeff_modulus_bit_sizes([60, 40, 40, 40, 40, 40, 40, 40, 60])
    .set_scale(2**40)
    .enable_bootstrapping(
        config=BootstrappingConfig(CtoS_piece=3, StoC_piece=3, taylor_number=11),
        boot_type=BootstrappingType.SLIM,
        policy=BootstrappingPolicy.AUTO,
    )
    .build()
    # build() calls generate_bootstrapping_params(), bootstrapping_key_indexs(),
    # merges shifts, and generates the Galois key — all transparently.
)

# 2. Build the model.
model = Sequential(
    Linear(128, 64), ApproxReLU(degree=3),   # consumes 2 levels
    Linear(64, 64),  ApproxReLU(degree=3),   # consumes 2 more, level now 3
    Linear(64, 64),  ApproxReLU(degree=3),   # after 2nd act, level=1 → AUTO boots here
    Linear(64, 10),
    ctx=ctx,
)

# 3. Load weights (not shown for brevity).

# 4. Encrypted inference.
enc_input = ctx.encrypt(plaintext_input)
enc_output = model(enc_input)   # bootstrapping fires automatically at level <= 1
result = enc_output.decrypt()
```

The bootstrapping call is invisible to the user. From the model's perspective, the
ciphertext simply has a restored level when it arrives at the third activation block.

---

## 4. Parameter Sizing with Bootstrapping

Sizing a context for bootstrapping requires accounting for two separate level budgets:
the levels that the bootstrapping circuit itself consumes, and the levels available to
the network between bootstraps.

### 4.1 How BootstrappingConfig Affects Level Consumption

The bootstrapping circuit's depth depends on `CtoS_piece`, `StoC_piece`, and
`taylor_number`:

- **CtoS_piece / StoC_piece (range 2-5):** Each DFT stage (CoeffToSlot and
  SlotToCoeff) consumes approximately `ceil(log(N/2) / piece)` levels. Smaller piece
  values mean fewer, deeper stages (more levels consumed per DFT). Larger values mean
  more stages but fewer levels each. Default of 3 is a reasonable middle ground.

- **taylor_number (range 6-15):** The EvalMod step evaluates a degree-`taylor_number`
  polynomial of a scaled sine function. The BSGS evaluation costs
  `ceil(log2(taylor_number))` levels. Default of 11 costs 4 levels.

- **less_key_mode:** Does not affect level consumption; reduces the Galois key count
  by ~30% at a 15-20% performance cost.

For default parameters (piece=3, taylor=11) with N=16384, the bootstrapping circuit
consumes approximately 8-10 levels. For slim bootstrapping, the cost is slightly lower
because one DFT stage runs before ModRaise rather than after.

### 4.2 Total Level Budget

The required chain length follows this formula:

```
total_usable_levels >= boot_circuit_levels + network_levels_per_segment + safety_margin
```

Where `network_levels_per_segment` is the level depth consumed between two consecutive
bootstrapping calls. With `AUTO` policy bootstrapping at level 1, the effective
levels usable by the network are `total_usable_levels - boot_circuit_levels - 1`.

### 4.3 Sizing Table

The table below uses default `BootstrappingConfig` (piece=3, taylor=11, regular
bootstrapping) consuming ~9 levels for the bootstrapping circuit.

| Network architecture | Network levels | Boot circuit levels | Total usable needed | Chain length | N | Recommended coeff_modulus_bit_sizes |
|---|---|---|---|---|---|---|
| Linear → Square → Linear | 1 | 9 | 11 | 13 | 32768 | `[60, 40x11, 60]` |
| 2x ApproxReLU(3) | 4 | 9 | 14 | 16 | 32768 | `[60, 40x14, 60]` |
| 3x ApproxReLU(3) | 6 | 9 | 16 | 18 | 32768 | `[60, 40x16, 60]` |
| 4x ApproxReLU(3) | 8 | 9 | 18 | 20 | 32768 | `[60, 40x18, 60]` |
| 2x ApproxReLU(5) | 6 | 9 | 16 | 18 | 32768 | `[60, 40x16, 60]` |

Notes:
- "Chain length" is `len(coeff_modulus_bit_sizes)`.
- Interior primes are written as `40xK` meaning K primes of 40 bits (matching a
  scale of 2^40).
- "Total usable" adds 1 as safety margin.
- N=32768 is required for chains of this length to stay within 128-bit security bounds.
  N=16384 supports up to about 7 usable levels at 128-bit security; that is sufficient
  only for slim bootstrapping with small networks.

### 4.4 Slim Bootstrapping Savings

Slim bootstrapping consumes approximately 6-7 levels (vs 8-10 for regular) because
the SlotToCoeff stage runs before ModRaise, outside the refreshed chain. For real-valued
networks, switching to slim bootstrapping allows using a shorter chain:

```python
# Slim bootstrapping for a 3x ApproxReLU(3) network:
ctx = (
    FHEContext()
    .set_poly_modulus_degree(32768)
    .set_coeff_modulus_bit_sizes([60] + [40] * 13 + [60])  # 13 interior = 15 chain
    .set_scale(2**40)
    .enable_bootstrapping(boot_type=BootstrappingType.SLIM)
    .build()
)
```

### 4.5 Multiple Bootstraps

If the network is deeper than what one bootstrap can cover (i.e., the network's total
level demand exceeds `total_usable_levels - boot_circuit_levels`), bootstrapping will
fire more than once during a single forward pass. The `AUTO` policy handles this
naturally — it bootstraps again whenever the level drops to 1, regardless of how many
times this has already happened during the same forward call. Each bootstrap costs
~100 ms on an RTX 4090. Users should size their chains to require as few bootstraps
as possible, as each one adds noise and latency.

---

## 5. FHEContext.build() Changes Needed

When `enable_bootstrapping()` was called before `build()`, the `build()` method must
perform additional setup after the standard key generation. The full `build()` sequence
with bootstrapping enabled is:

**Existing steps (always performed):**

1. Validate that `poly_modulus_degree`, `coeff_modulus_bit_sizes`, and `scale` are set.
2. Call `create_ckks_context_with_security()` to produce the `CKKSContextPtr`.
3. Instantiate `CKKSEncoder`, `CKKSOperator`, `CKKSKeyGenerator`.
4. Generate secret key, public key, and relinearization key.
5. Generate a `CKKSGaloiskey` with the shifts required by the network's linear layers
   (collected at `build()` time or deferred until first `forward()` call — the current
   architecture determines which; this document assumes build-time collection is not
   yet implemented and that the Galois key for the network is handled separately).

**New steps when `self._bootstrapping_enabled is True`:**

6. Call `operator.generate_bootstrapping_params(self._scale, self._boot_config, self._boot_type)`.
   This precomputes the DFT matrices and EvalMod polynomial on the GPU. It may take
   several seconds on first call (one-time cost).

7. Call `boot_shifts = operator.bootstrapping_key_indexs()` to retrieve the required
   Galois shift indices. The exact set depends on `poly_modulus_degree`, `CtoS_piece`,
   and `StoC_piece`.

8. Collect `network_shifts` from the registered `Sequential` model, if any. If no
   model has been registered at `build()` time, `network_shifts` is empty and the
   user is responsible for ensuring the Galois key covers any rotation shifts their
   model will need. A future SDK version could accept the model at context build time.

9. Compute `all_shifts = list(set(boot_shifts + network_shifts))`.

10. Construct `gk = CKKSGaloiskey(context, all_shifts)` using the exact-shift
    constructor.

11. Call `keygen.generate_galois_key(gk, sk)` to fill the Galois key.

12. Store `self._boot_gk = gk`. The relinearization key `self._rk` generated in step 4
    is already available and is reused for bootstrapping.

The complete state stored on `FHEContext` after a bootstrapping-enabled `build()`:

```python
self._bootstrapping_enabled: bool = True
self._boot_config: BootstrappingConfig       # from enable_bootstrapping()
self._boot_type: BootstrappingType           # from enable_bootstrapping()
self._boot_policy: BootstrappingPolicy       # from enable_bootstrapping()
self._boot_gk: CKKSGaloiskey                 # generated in build(), step 10-11
# self._rk: CKKSRelinkey                     # already present from standard build
```

The `_boot_gk` and `_rk` are what `ctx.bootstrap(ct)` will pass to the backend
operator at inference time.

### 5.1 Error Cases in build()

- If `generate_bootstrapping_params` raises (e.g., chain is too short for the
  bootstrapping circuit), `build()` must re-raise as a `ValueError` with a message
  that includes the chain length, the minimum required length, and a hint to increase
  `poly_modulus_degree`.

- If `bootstrapping_key_indexs()` raises because `generate_bootstrapping_params` was
  not called (a programming error in this code, not a user error), the exception
  should propagate unmodified as a `RuntimeError`.

---

## 6. User-Facing ctx.bootstrap() Method

`FHEContext.bootstrap()` is the public method through which both `Sequential` and
advanced users invoke bootstrapping.

```python
def bootstrap(self, ct: "Ciphertext") -> "Ciphertext":
    """Refresh ct back to near-full depth using HEonGPU bootstrapping.

    Args:
        ct: The ciphertext to refresh. Must be at level 0 (or level <= 1 for
            AUTO policy use). The input ciphertext is not modified.

    Returns:
        A new Ciphertext at near-full depth, with slightly higher noise than
        a freshly encrypted ciphertext.

    Raises:
        RuntimeError: If enable_bootstrapping() was not called before build().
        RuntimeError: If ct was not produced by this context.
        ValueError: If ct is not at the expected level for bootstrapping.
    """
    if not self._bootstrapping_enabled:
        raise RuntimeError(
            "Bootstrapping was not enabled for this context. "
            "Call enable_bootstrapping() before build()."
        )
    if ct._context is not self:
        raise RuntimeError(
            "Ciphertext belongs to a different FHEContext."
        )

    raw_ct = ct._raw
    if self._boot_type is BootstrappingType.SLIM:
        raw_refreshed = self._operator.slim_bootstrapping(
            raw_ct, self._boot_gk, self._rk
        )
    else:
        raw_refreshed = self._operator.regular_bootstrapping(
            raw_ct, self._boot_gk, self._rk
        )

    return Ciphertext._from_raw(raw_refreshed, context=self)
```

Key design decisions made explicit here:

**Returns a new Ciphertext, does not modify in-place.** The HEonGPU backend functions
(`regular_bootstrapping`, `slim_bootstrapping`) return new `CKKSCiphertext` objects;
the input is not modified. The Python method mirrors this by returning a new `Ciphertext`
wrapper. This makes the call site in `Sequential._maybe_bootstrap` correct:
`x = self._ctx.bootstrap(x)` reassigns `x` to the refreshed ciphertext while the old
one is naturally garbage-collected.

**Level precondition.** The `regular_bootstrapping` backend requires the input to be
at level 0. However, the SDK's `AUTO` policy calls `bootstrap()` when `ct.level <= 1`.
If `ct.level == 1`, the backend will receive a ciphertext that is not at level 0. The
implementation should perform a `mod_drop_inplace` on the raw ciphertext to bring it
to level 0 before calling the bootstrapping operator:

```python
if raw_ct.level > 0:
    self._operator.mod_drop_inplace(raw_ct_copy)
raw_refreshed = self._operator.regular_bootstrapping(raw_ct_copy, ...)
```

Since `mod_drop_inplace` modifies in-place and we must not modify `ct`, the
implementation should work on `raw_ct.copy()`.

**Accuracy impact.** The bootstrapped ciphertext carries the error of the original
computation plus a bootstrapping error. The bootstrapping error comes from two sources:
the truncated Taylor series in EvalMod (controlled by `taylor_number`; higher values
reduce this error) and the floating-point error in the precomputed DFT matrices. For
inference workloads, this additional error is typically on the order of 2^-20 to 2^-30
in absolute terms at scale 2^40, which is negligible relative to the model's
approximation error from polynomial activation functions. However, bootstrapping
cannot be applied an unbounded number of times on a single ciphertext without
eventually accumulating significant error. In practice, for inference on fixed inputs,
this is not a concern; each input is bootstrapped at most a few times per forward pass.

---

## 7. Limitations and Known Constraints

### 7.1 poly_modulus_degree Requirement

Bootstrapping requires a longer modulus chain than plain inference. The bootstrapping
circuit alone consumes 8-10 levels with default parameters. This forces `N` to
32768 for most non-trivial networks. At N=32768, key generation takes longer and keys
occupy significantly more GPU memory than at N=8192. The Galois key for bootstrapping
can be several GB; `less_key_mode=True` in `BootstrappingConfig` should be considered
for GPUs with less than 24 GB of VRAM.

### 7.2 Context is Immutable After build()

Bootstrapping setup cannot be added to an already-built context. `enable_bootstrapping()`
raises `RuntimeError` if called after `build()`. If a user discovers mid-development
that their network requires bootstrapping, they must create a new `FHEContext` with a
longer chain and `enable_bootstrapping()` called, then re-encrypt their inputs. There
is no in-place upgrade path.

### 7.3 Level 0 Precondition

`regular_bootstrapping` requires the input ciphertext to be at level 0. Calling it on
a ciphertext with `level > 0` is undefined behavior at the HEonGPU layer (it may
crash or produce garbage). The `ctx.bootstrap()` implementation guards against this
with the `mod_drop_inplace` normalization described in Section 6. Callers using the
raw `_backend` bindings directly must manage this themselves.

### 7.4 Galois Key Must Cover All Required Shifts

If the `Sequential` model uses a `Linear` layer whose diagonal method requires a
rotation shift that was not included in `all_shifts` at `build()` time, the rotation
will raise a C++ exception at runtime (the Galois key simply doesn't have a key for
that shift). This error manifests as a `RuntimeError` in Python with a message about
a missing Galois element. The fix is to register the model with the context before
calling `build()`, or to call `build()` only after the model's rotation requirements
are known. A `FHEContext.register_model(model)` API that collects required shifts is a
planned future addition.

### 7.5 BootstrappingConfigV2 Not Yet Exposed

HEonGPU supports a `BootstrappingConfigV2` for the `regular_bootstrapping_v2` variant
(based on IACR 2020/1203, Han-Ki non-sparse key bootstrapping). This config supports
Chebyshev polynomial approximation for EvalMod (via `PolyType.CHEBYSHEV`) which offers
better numerical stability than Taylor series, and additional parameters for controlling
the sine approximation range (`K` parameter) and double-angle iterations. The `_backend`
binding for `regular_bootstrapping_v2` is present in `bind_operator.cu` but
`BootstrappingConfigV2` is not yet bound. The `FHEContext` high-level API exposes only
`BootstrappingType.REGULAR` (maps to `regular_bootstrapping`) and
`BootstrappingType.SLIM` (maps to `slim_bootstrapping`). The v2 path requires additional
switch key management (dense-to-sparse and sparse-to-dense switch keys) that has no
Python binding at this time.

### 7.6 Bit and Gate Bootstrapping Are Out of Scope

HEonGPU also implements bit bootstrapping and gate bootstrapping for binary-encoded
messages. These require specific constraints on the last modulus prime (`q_0 = 2*scale`
or `q_0 = 3*scale`) and are intended for binary circuits, not neural network inference.
They are not exposed in the SDK and are not part of the `BootstrappingType` enum.
