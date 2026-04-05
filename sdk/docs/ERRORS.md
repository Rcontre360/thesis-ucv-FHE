# SDK Documentation — Error Audit

**Audit date:** 2026-04-05  
**Files audited:** `sdk/docs/API.md`, `sdk/README.md`, `sdk/docs/RESEARCH_SUMMARY.md`, `sdk/src/backend/bindings.cu`  
**Ground truth:** HEonGPU headers (`context.cuh`, `operator.cuh`, `keygenerator.cuh`, `secstdparams.h`, `defines.h`) and example `2_basic_ckks.cpp`

Severity: **CRITICAL** = wrong/non-compiling implementation if followed literally · **MODERATE** = subtle bugs or maintenance confusion · **MINOR** = cosmetic/incomplete

---

## ERROR-001 — CRITICAL: `set_coeff_modulus_bit_sizes` has the wrong signature

**Files:** `API.md` (Configuration setters), `README.md` (all parameter examples)

**Wrong:** `set_coeff_modulus_bit_sizes(bit_sizes: list[int])` — single flat list, e.g. `[60, 40, 40, 60]`.

**Actual HEonGPU signature:**
```cpp
void set_coeff_modulus_bit_sizes(
    const std::vector<int>& log_Q_bases_bit_sizes,   // ciphertext modulus chain
    const std::vector<int>& log_P_bases_bit_sizes);  // key-switching auxiliary modulus
```
Source: `context.cu` lines 54–56. Every HEonGPU example uses two vectors, e.g. `set_coeff_modulus_bit_sizes({60, 30, 30, 30}, {60})`.

**Correct behavior:** The Python API must accept two separate lists, or accept one flat list and document an explicit Q/P split rule (e.g. "the last element is P"). The split rule must appear in every example. Without this, any implementation calling the single-vector form will not compile.

---

## ERROR-002 — CRITICAL: `set_keyswitching_type` does not exist in HEonGPU

**Files:** `API.md` (Configuration setters), `README.md`

**Wrong:** `FHEContext.set_keyswitching_type(ktype: KeyswitchingType)` is listed as a valid fluent setter.

**Actual HEonGPU behavior:** No such method exists on `HEContextImpl<Scheme::CKKS>`. The keyswitching type is derived automatically from the P vector size:
```cpp
keyswitching_type_ = (log_P_bases_bit_sizes.size() == 1)
    ? keyswitching_type::KEYSWITCHING_METHOD_I
    : keyswitching_type::KEYSWITCHING_METHOD_II;
```
Source: `context.cu` lines 67–70.

**Correct behavior:** Remove `set_keyswitching_type` from the public API. Document that METHOD_I is selected by one P prime, METHOD_II by multiple P primes. If the setter is kept as a Python-layer convenience, document the internal translation to P vector size.

---

## ERROR-003 — CRITICAL: `set_scale` does not exist in HEonGPU

**Files:** `API.md` (Configuration setters), `README.md`

**Wrong:** `FHEContext.set_scale(scale: float)` is listed as a valid setter that globally configures the encoding scale.

**Actual HEonGPU behavior:** No `set_scale` method on `HEContextImpl<Scheme::CKKS>`. Scale is a per-call argument to the encoder:
```cpp
encoder.encode(plain, message, scale);
```
Source: `encoder.cuh` lines 55–59.

**Correct behavior:** `FHEContext` must store scale as a plain Python attribute and inject it into every `CKKSEncoder.encode` call internally. The docs must state that `set_scale` is a Python-layer convenience, not a HEonGPU context method.

---

## ERROR-004 — CRITICAL: `build()` does not correspond to any HEonGPU method, and does not generate keys

**Files:** `API.md` (`build()` section), `README.md`

**Wrong:** (1) The HEonGPU method is `generate()`, not `build()`. (2) `build()` is said to produce all keys — but `generate()` only creates the mathematical context (NTT tables, prime vectors). Key generation is a fully separate step:
```cpp
HEKeyGenerator<Scheme::CKKS> keygen(context);
keygen.generate_secret_key(secret_key);
keygen.generate_public_key(public_key, secret_key);
keygen.generate_relin_key(relin_key, secret_key);
```
Source: `2_basic_ckks.cpp` lines 62–69.

**Correct behavior:** Python `build()` must orchestrate: call `context->generate()`, create a `CKKSKeyGenerator`, generate and store sk/pk/rlk, then create `CKKSEncoder`, `CKKSEncryptor`, `CKKSDecryptor`, and `CKKSOperator` instances. All of steps 2–4 are undocumented.

---

## ERROR-005 — CRITICAL: `SecurityLevel.NONE` is exposed in bindings but the docs say it is not

**Files:** `API.md` (Enums): "`NONE` is not exposed."

**Actual `bindings.cu` lines 33–37:**
```cpp
py::enum_<sec_level_type>(m, "SecurityLevel")
    .value("NONE", sec_level_type::none)   // explicitly bound
    ...
```

**Correct behavior:** Either remove `NONE` from `bindings.cu`, or update the docs to document it with a warning. As-is, Python code can import and use `SecurityLevel.NONE` despite the docs claiming it doesn't exist.

---

## ERROR-006 — CRITICAL: `KeyswitchingType.NONE` is exposed but undocumented; the whole enum is a dead abstraction

**Files:** `API.md` (Enums), `bindings.cu` lines 39–43

`bindings.cu` exposes `KeyswitchingType.NONE` which is not in the API docs. More fundamentally, since `set_keyswitching_type` does not exist (ERROR-002), the `KeyswitchingType` enum cannot serve as a configuration mechanism. Using `NONE` would leave the context's internal keyswitching type unset, causing operations to throw `std::invalid_argument("Invalid Key Switching Type")`.

**Correct behavior:** Fix ERROR-002 first. Then either remove `NONE` from the binding or mark it as internal/reserved.

---

## ERROR-007 — CRITICAL: `FHEContext.default()` parameters exceed HEonGPU's security limit and will throw at runtime

**Files:** `API.md` (`FHEContext.default()`), `README.md` (Depth Budget Guide, Crypto Parameter table), `RESEARCH_SUMMARY.md` (§7)

**Wrong:** Default `[60, 40, 40, 60]` described as "200 total bits" within the 128-bit security limit for N=8192.

**Actual HEonGPU limit:** `heongpu_128bit_std_parms(8192) = 218` bits. Source: `secstdparams.h` lines 29–32. The security check applies to `sum(Q) + sum(P)`. If `[60, 40, 40, 60]` is naively used as a four-element Q, the sum is already 200 bits — adding any P prime pushes the total over 218, causing a `std::runtime_error` at runtime.

The only consistent interpretation is `Q={60,40,40}, P={60}` (total = 200 bits < 218). This gives 2 usable levels and matches the rest of the docs — but **the Q/P split rule is never stated anywhere**. An implementer who treats all four values as Q will get a runtime error.

Additionally, RESEARCH_SUMMARY.md states the 128-bit limit as "200 bits" — the actual HEonGPU value is **218 bits** (the 200-bit figure comes from SEAL's different baseline).

**Correct behavior:** Document the Q/P split explicitly. Fix the security constant to 218 bits. Update `default()` to show `set_coeff_modulus_bit_sizes({60, 40, 40}, {60})`.

---

## ERROR-008 — CRITICAL: `set_security_level` does not correspond to a HEonGPU method

**Files:** `API.md` (Configuration setters), `README.md`

**Wrong:** `FHEContext.set_security_level(level: SecurityLevel)` is listed as a fluent setter.

**Actual HEonGPU behavior:** Security level is a constructor argument:
```cpp
HEContextImpl(const sec_level_type = sec_level_type::sec128);
```
Source: `context.cuh` line 47. There is no post-construction setter.

**Correct behavior:** Store the security level in the Python object and pass it to `GenHEContext<Scheme::CKKS>(sec_level)` when `build()` creates the C++ context. This implementation detail must be documented.

---

## ERROR-009 — CRITICAL: `Linear` claiming "0 levels" is wrong; it consumes 1 level

**Files:** `API.md` (operation depth table `W @ ct` row), `README.md` (level cost table)

**Wrong:** "Linear — 0 levels — matrix-vector multiply uses plaintext weights."

**Correct analysis:** The Halevi-Shoup diagonal method calls `multiply_plain_inplace` for each diagonal, which sets `rescale_required_ = true` (confirmed: `operator.cuh` line 767). A single `rescale_inplace` must be called after accumulation, consuming one modulus level. The same table correctly says `ct * Plaintext` costs 1 level — `W @ ct` is composed of such operations and costs **1 level total**.

**Cascading effect:** The depth budget examples are wrong. `Sequential(Linear, Square, Linear)` costs 1+1+1=**3 levels**, not 1. The default chain `[60,40,40,60]` only supports 2 usable levels — insufficient for this model.

**Contradiction within the same docs:** `API.md` operation table says `W @ ct` costs 1 level; `README.md` layer table says `Linear` costs 0. These directly contradict each other.

---

## ERROR-010 — CRITICAL: Rescaling and relinearization are NOT automatic in HEonGPU

**Files:** `API.md` (depth table note): "Relinearization and rescaling are automatic. Users never call them explicitly."

**Actual HEonGPU behavior:** After `multiply_inplace`, HEonGPU sets flags but does NOT relinearize or rescale. The example code is unambiguous:
```cpp
operators.multiply_inplace(C1, C1);
operators.relinearize_inplace(C1, relin_key);  // explicit
operators.rescale_inplace(C1);                  // explicit
```
Source: `2_basic_ckks.cpp` lines 121–126. Attempting any operation on a ciphertext where `relinearization_required_ = true` throws `std::invalid_argument`.

**Correct behavior:** Python `Ciphertext.__mul__` must explicitly call `multiply_inplace` → `relinearize_inplace` → `rescale_inplace`. Python `ct * Plaintext` must explicitly call `multiply_plain_inplace` → `rescale_inplace`. The docs must document this sequence, not claim it is automatic.

---

## ERROR-011 — CRITICAL: `build()` generates no Galois key; `Linear` and `dot` require one; no generation path is documented

**Files:** `API.md` (`build()`, `Linear`, `Ciphertext.dot()`), `README.md`

`build()` is said to produce sk/pk/rlk. No Galois key is mentioned. But `rotate_rows_inplace(ct, galois_key, shift)` requires a `CKKSGaloiskey` as a mandatory argument — there is no default. Without it, neither `Linear.forward()` nor `ct.dot(ct)` can be implemented.

Deeper problem: the required rotation shifts for the diagonal method are `{0, 1, ..., in_features-1}`, which depend on layer architecture unknown at `build()` time. No path is documented for requesting specific rotation keys, nor is there any hook in `Linear` or `FHEContext` to generate them on demand.

**Correct behavior:** Either (a) require `ctx.generate_galois_keys(shifts)` before using `Linear`, documenting the required shifts; or (b) have `Linear` lazily generate keys on first `forward()` call (requiring a `FHEContext` back-reference on `Linear`). Either choice must be documented.

---

## ERROR-012 — CRITICAL: `bindings.cu` is missing 10 bindings required to implement the public API

**Files:** `sdk/src/backend/bindings.cu`, `RESEARCH_SUMMARY.md` §5

| Missing binding | Required for | Exists in HEonGPU? |
|---|---|---|
| `CKKSGaloiskey` class | Any rotation | Yes |
| `generate_galois_key(gk, sk)` | Rotation key generation | Yes |
| `rotate_rows_inplace(ct, gk, shift)` | `W @ ct`, `ct.dot(ct)` | Yes |
| `add_plain_inplace(ct, pt)` | `ct + Plaintext` | Yes |
| `sub_inplace(ct, ct)` | `ct - ct` | Yes |
| `sub_plain_inplace(ct, pt)` | `ct - Plaintext` | Yes |
| `multiply_plain_inplace(ct, pt)` | `ct * Plaintext`, diagonal method | Yes |
| `negate_inplace(ct)` | `plain - ct` reflected subtraction | Yes |
| `mod_drop_inplace(pt)` | Level-matching plaintext before any ct+pt/ct*pt | Yes |
| `mod_drop_inplace(ct)` | Level-matching ciphertext | Yes |

`mod_drop_inplace` is especially critical: `add_plain` and `multiply_plain` both throw `std::logic_error("Ciphertexts leveled are not equal")` if `ciphertext.depth != plaintext.depth` (confirmed: `operator.cuh` lines 202, 757). Every freshly encoded plaintext is at depth=0; after one rescale the ciphertext is at depth=1; without `mod_drop` the operation throws.

---

## ERROR-013 — CRITICAL: `Plaintext.decode()` requires a context back-reference that is never specified

**Files:** `API.md` (Plaintext `decode()`)

`decode()` is documented as "Shorthand for `context.decode(self)`" — but `Plaintext` has no documented `context` attribute. The back-reference to `FHEContext` is never mentioned as a design requirement. Arithmetic operations on `Plaintext` objects must propagate this reference; the propagation rule is unspecified. Mixing `Plaintext` objects from different contexts must raise an error, but the detection mechanism is also unspecified.

---

## ERROR-014 — CRITICAL: `Ciphertext.decrypt()` requires a context back-reference that is never specified

**Files:** `API.md` (Ciphertext `decrypt()`)

Same structural problem as ERROR-013. `Ciphertext` must store a back-reference to its `FHEContext`. The `FHEContext.decrypt()` spec says it raises `ValueError` for ciphertexts from other contexts — but the detection mechanism (identity comparison of stored references) is never documented. All ciphertexts from arithmetic operations must propagate this back-reference; the rule is unspecified.

---

## ERROR-015 — CRITICAL: `Plaintext.__mul__(Plaintext)` is cryptographically incoherent

**Files:** `API.md` (Plaintext arithmetic operators)

`Plaintext * Plaintext` is specified as returning a `Plaintext`. In HEonGPU's model a `CKKSPlaintext` is an RNS-NTT polynomial at scale Δ. Multiplying two produces a polynomial at scale Δ² — incompatible with any ciphertext at scale Δ, and there is no `rescale` for plaintexts. HEonGPU has no `multiply_plain_plain` operation. This operator either must be removed, or replaced with a note that it decodes both plaintexts to `list[float]`, multiplies element-wise in Python, and re-encodes — making it a pure Python operation with no HEonGPU involvement.

---

## ERROR-016 — MODERATE: `ct + Plaintext` and `ct * Plaintext` silently throw unless levels are matched first

**Files:** `API.md` (Ciphertext operators), `README.md` (quickstart)

HEonGPU checks `ciphertext.depth_ == plaintext.depth_` before every `add_plain` and `multiply_plain`, throwing `std::logic_error` on mismatch. A freshly encoded `Plaintext` is always at depth=0. After one rescale a `Ciphertext` is at depth=1. All quickstart examples showing `ct + pt` or `ct * pt` will break in any multi-level computation. The Python layer must call `mod_drop_plain_inplace(pt)` for each level of difference before every such operation. This constraint is never mentioned anywhere in the docs.

---

## ERROR-017 — MODERATE: BSGS formula attributed imprecisely; "BSGS" is non-standard FHE terminology

**Files:** `API.md` (activation depth note), `RESEARCH_SUMMARY.md` §4

The formula `ceil(log2(d))` is numerically correct for d∈{2,3,5} but the algorithm should be called **Paterson-Stockmeyer**, not BSGS (Baby-step Giant-step is a group-theory algorithm). The description should specify: "Paterson-Stockmeyer polynomial evaluation achieves depth `ceil(log2(d))` for degree-d polynomials with baby-step size `floor(sqrt(d))`." Using "BSGS" without qualification is non-standard in the FHE literature and will confuse implementers who search for it.

---

## ERROR-018 — MODERATE: Operation depth table is missing `ct - Plaintext` and `ct - list[float]` rows

**Files:** `API.md` (operation depth table)

`ct + Plaintext` (0 levels) and `ct + list[float]` (0 levels) are present. The corresponding subtraction rows `ct - Plaintext` and `ct - list[float]` (also 0 levels) are absent. Incomplete table.

---

## ERROR-019 — MODERATE: `ApproxReLU`/`ApproxSigmoid` "degree must be odd" stated as hard constraint without justification

**Files:** `API.md` (ApproxReLU/ApproxSigmoid parameter tables)

"Must be odd and ≥ 3" is presented as a hard restriction. It is an approximation quality recommendation (odd polynomials better approximate antisymmetric functions over symmetric intervals), not a cryptographic or algorithmic requirement. Even-degree inputs to Paterson-Stockmeyer are valid and compute correctly. Stating it as a constraint will cause unnecessary `ValueError` raises. Correct statement: "Odd degree recommended for better approximation quality; even degrees are permitted but provide higher approximation error near the boundary."

---

## ERROR-020 — MODERATE: `FHEContext.decode()` and `Plaintext.decode()` create a circular dependency

**Files:** `API.md` (`FHEContext.decode()`, `Plaintext.decode()`)

`Plaintext.decode()` is "Shorthand for `context.decode(self)`" which calls `FHEContext.decode(plaintext)`. The spec creates a circular structure without designating a primary implementation. Combined with ERROR-013 (missing context back-reference), neither method can be implemented without first resolving the context-reference problem. The docs should designate one as canonical and describe the back-reference storage.

---

## ERROR-021 — MODERATE: Security bit-limit stated as 200 bits for N=8192; actual HEonGPU limit is 218 bits

**Files:** `RESEARCH_SUMMARY.md` (§1 and §7)

`RESEARCH_SUMMARY.md` states the 128-bit limit for N=8192 as 200 bits. The actual HEonGPU value from `secstdparams.h` is **218 bits**. The 200-bit figure comes from Microsoft SEAL's parameter tables (Gaussian secret distribution). HEonGPU uses ternary secrets, yielding a slightly higher bound. Any validation code enforcing 200 bits will incorrectly reject valid 201–218-bit configurations.

---

## ERROR-022 — MODERATE: Default Galois key generation covers only powers-of-2 shifts; diagonal method needs all `{1..in_features-1}`

**Files:** `RESEARCH_SUMMARY.md` §3

The default `Galoiskey` constructor generates rotation keys for powers-of-2 only, up to `2^MAX_SHIFT=8` (i.e., shifts ±1, ±2, ±4, ..., ±256). Source: `defines.h` line 28. The diagonal method for a `Linear(64, *)` layer needs all shifts `{0,1,...,63}` — most of which are not powers of 2. The `Galoiskey(context, shifts)` constructor with an explicit shift list is required. The docs must specify the exact shifts needed for each layer size and document how to request them.

---

## ERROR-023 — MODERATE: Diagonal method pseudocode assumes square matrix (out=in)

**Files:** `RESEARCH_SUMMARY.md` §3

The pseudocode loop `for k in 0..in-1` only works for square W (out==in). `Linear` allows `out_features ≠ in_features`. For rectangular matrices, Halevi-Shoup requires zero-padding W to square or a modified diagonal extraction. An implementer following this pseudocode will produce incorrect results for non-square layers.

---

## ERROR-024 — MINOR: `Ciphertext.size` semantics are ambiguous after `W @ ct`

**Files:** `API.md` (Ciphertext `size` property)

After `W @ ct` where `out_features ≠ in_features`, `size` per the docs returns the original slot count, not `out_features`. It's unclear whether `size` means "ring capacity" (always N/2) or "logically meaningful slots" (changes after matmul). The docs must choose one definition and state it precisely.

---

## ERROR-025 — MINOR: `FHEContext.decrypt()` return length claim is wrong for dimension-reducing ciphertexts

**Files:** `API.md` (`FHEContext.decrypt()`)

"Returns a `list[float]` of the same length as the original plaintext" — after `W @ ct` where `out_features < in_features`, `decrypt()` returns N/2 values, not `out_features`. Correct statement: "Returns a `list[float]` of length N/2; slots beyond the logically meaningful output size contain noise and should be discarded."

---

## ERROR-026 — MINOR: `ct * float` (scalar) depth cost is absent from the operation table

**Files:** `README.md` (quickstart: `g = a * 2.0`), `API.md` (depth table)

`ct * float` appears in the quickstart but has no entry in the depth table. It should be documented (1 level, same as `ct * Plaintext` since the scalar is auto-encoded).

---

## ERROR-027 — MINOR: `RESEARCH_SUMMARY.md` says `add_plain_inplace` "needs adding" — it exists in HEonGPU already

**Files:** `RESEARCH_SUMMARY.md` §5

`add_plain_inplace` exists in `operator.cuh` line 265. It needs a **binding in `bindings.cu`**, not creation in HEonGPU. Same for `sub_inplace`, `multiply_plain_inplace`, `negate_inplace`. The phrasing "needs adding" is misleading — should say "needs binding."

---

## ERROR-028 — MINOR: README and API.md depth tables directly contradict on `Linear`

**Files:** `README.md` (Depth Budget Guide), `API.md` (depth table)

`README.md`: Linear = 0 levels. `API.md` `W @ ct` row: 1 level. `Linear.forward` = `W @ ct + b`. Therefore Linear = 1 level. The README entry is wrong. (Full analysis in ERROR-009.)

---

## ERROR-029 — CRITICAL: `multiply()` throws if inputs have `rescale_required_` or `relinearization_required_` set

**Files:** `API.md` (Ciphertext arithmetic operators), `RESEARCH_SUMMARY.md` §2

Neither document mentions that `multiply(ct1, ct2)` raises `std::invalid_argument` if either input has `rescale_required_ = true` (operator.cuh lines 651–657: "Ciphertexts can not be multiplied because of the noise!") or `relinearization_required_ = true` (lines 637–643). These are hard blocking preconditions. Any chained multiplication that skips `relinearize_inplace` + `rescale_inplace` will crash at the C++ level. This affects every multi-layer network.

---

## ERROR-030 — CRITICAL: `ct + ct` also has preconditions — throws on depth mismatch or mismatched `relinearization_required_` flags

**Files:** `API.md` (depth table note "ct + ct → 0 levels, homomorphic addition"), `RESEARCH_SUMMARY.md` §2

Addition is claimed unconditionally free. In HEonGPU `add(ct1, ct2)` (operator.cu lines 70–85) throws if the two ciphertexts have different `depth_` values or different `relinearization_required_` flags. This means any residual addition between a rescaled ciphertext and a fresh ciphertext (e.g. adding a skip-connection from an earlier layer) requires explicit level-alignment via `mod_drop_inplace` before the add. The "0 levels, no preconditions" characterisation is wrong.

---

## ERROR-031 — MODERATE: `rescale_required` property docstring hides that `multiply()` hard-blocks on it while `multiply_plain` does not

**Files:** `bindings.cu` lines 204–208

The `rescale_required` docstring reads "True if rescale_inplace must be called." It hides that `multiply()` raises an exception when this flag is set on either input, whereas `multiply_plain` proceeds silently and compounds the scale (Δ → Δ²). This asymmetric enforcement across the two multiply variants is entirely undocumented and will cause silent scale corruption whenever a plaintext multiply is chained without an intervening rescale. Should read: "True after any multiplication. A subsequent ct*ct will throw until cleared by `rescale_inplace`; `ct * Plaintext` proceeds but compounds the encoding scale."

---

## ERROR-032 — MINOR: `bindings.cu` comment "usable = get_ciphertext_modulus_count() - 2" is arithmetically wrong

**Files:** `bindings.cu` lines 91–93

The comment states "usable levels = `get_ciphertext_modulus_count() - 2`." With the flat-list splitter assigning Q = `[60, 40, 40]` (Q_size = 3) and P = `[60]`, `get_ciphertext_modulus_count()` returns Q_size = 3, giving 3 − 2 = **1** — but the correct answer is **2** (the two non-base Q primes are both consumable). The correct formula relative to Q alone is `get_ciphertext_modulus_count() - 1`. The README formula `len(flat_list) - 2` = 4 − 2 = 2 is correct because it counts Q + P entries.

---

## ERROR-033 — MINOR: `RESEARCH_SUMMARY.md` §5 "What bindings.cu is missing" is entirely stale

**Files:** `RESEARCH_SUMMARY.md` §5

All seven items listed as "must add" are already present in the current `bindings.cu`: `CKKSGaloiskey` (line 126), `generate_galois_key` (line 161), `rotate_rows_inplace` (line 376), out-of-place `rotate_rows` (line 393), `add_plain_inplace` (line 314), `multiply_plain_inplace` (line 329), `sub_inplace` (line 293), `sub_plain_inplace` (line 321), `negate_inplace` (line 300). The section is a stale pre-implementation to-do list that was never removed and now misrepresents the backend's actual state.

---

## Summary

| ID | Severity | Core issue |
|---|---|---|
| ERROR-001 | CRITICAL | `set_coeff_modulus_bit_sizes` takes two vectors (Q and P), not one flat list |
| ERROR-002 | CRITICAL | `set_keyswitching_type` does not exist; keyswitching is inferred from P vector size |
| ERROR-003 | CRITICAL | `set_scale` does not exist; scale is a per-encode argument |
| ERROR-004 | CRITICAL | HEonGPU method is `generate()` not `build()`; key gen is a separate step |
| ERROR-005 | CRITICAL | `SecurityLevel.NONE` is bound in `bindings.cu` despite docs claiming it's hidden |
| ERROR-006 | CRITICAL | `KeyswitchingType.NONE` is bound and undocumented; enum is a dead abstraction |
| ERROR-007 | CRITICAL | Default params exceed HEonGPU's 218-bit limit (not 200) for N=8192; Q/P split undocumented |
| ERROR-008 | CRITICAL | `set_security_level` doesn't exist; security level is a constructor argument |
| ERROR-009 | CRITICAL | `Linear` costs 1 level, not 0; all depth budget examples are wrong |
| ERROR-010 | CRITICAL | Rescale/relin are not automatic; must be called explicitly after every multiply |
| ERROR-011 | CRITICAL | Galois key generation is undocumented; `Linear` and `dot` cannot work without it |
| ERROR-012 | CRITICAL | 10 bindings missing from `bindings.cu` (rotations, plaintext ops, mod_drop, sub, negate) |
| ERROR-013 | CRITICAL | `Plaintext.decode()` needs a context back-reference; never specified |
| ERROR-014 | CRITICAL | `Ciphertext.decrypt()` needs a context back-reference; never specified |
| ERROR-015 | CRITICAL | `Plaintext * Plaintext` is incoherent; no HEonGPU backend operation exists for it |
| ERROR-029 | CRITICAL | `multiply()` throws if inputs have `rescale_required_` or `relinearization_required_` set |
| ERROR-030 | CRITICAL | `ct + ct` throws on depth mismatch or mismatched `relinearization_required_` flags |
| ERROR-016 | MODERATE | `ct + pt`/`ct * pt` require `mod_drop_plain` for level-matching; never mentioned |
| ERROR-017 | MODERATE | Algorithm is Paterson-Stockmeyer, not "BSGS"; depth formula attribution imprecise |
| ERROR-018 | MODERATE | Depth table missing `ct - Plaintext` and `ct - list[float]` rows |
| ERROR-019 | MODERATE | "Odd degree required" is a recommendation, not a hard constraint |
| ERROR-020 | MODERATE | `decode()` circular dependency between `FHEContext` and `Plaintext` |
| ERROR-021 | MODERATE | Security bit limit is 218 bits for N=8192 (not 200 as stated) |
| ERROR-022 | MODERATE | Default Galois key only covers powers-of-2 shifts; diagonal method needs all `{1..in_features-1}` |
| ERROR-023 | MODERATE | Diagonal method pseudocode only works for square matrices |
| ERROR-031 | MODERATE | `rescale_required` docstring hides asymmetric blocking behaviour vs `multiply_plain` |
| ERROR-024 | MINOR | `Ciphertext.size` semantics ambiguous after dimension-reducing operations |
| ERROR-025 | MINOR | `decrypt()` returns N/2 values, not `out_features` after matmul |
| ERROR-026 | MINOR | `ct * float` depth cost absent from operation table |
| ERROR-027 | MINOR | `add_plain_inplace` phrased as "needs adding" — it exists, it just needs binding |
| ERROR-028 | MINOR | README says Linear=0 levels, API.md says `W@ct`=1 level — direct contradiction |
| ERROR-032 | MINOR | `bindings.cu` usable-level comment formula is wrong (`-2` should be `-1`) |
| ERROR-033 | MINOR | `RESEARCH_SUMMARY.md` §5 to-do list is entirely stale; all items already done |

**17 CRITICAL · 9 MODERATE · 7 MINOR — 33 errors total**
