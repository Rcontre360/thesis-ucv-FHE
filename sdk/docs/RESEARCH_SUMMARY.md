# CKKS Research Summary

Synthesized from: original CKKS paper (Cheon 2017), HETAL (ICML 2023), Taiyi hardware accelerator, Llama-2-7B CKKS inference (CryptoLab 2026), high-precision bootstrapping (CCS 2025), CPA-D security analysis (CRYPTO 2024), thesis chapters 1–3, HEonGPU README, and existing bindings.

---

## 1. CKKS Core Concepts

### Encoding (Canonical Embedding)

- Ring: `R = Z[x] / <x^N + 1>`, N = power of 2
- Each ciphertext holds **N/2 complex slots** (or **N real slots** using conjugate-invariant subspace)
- Encoding = scale values by `Δ`, apply inverse canonical embedding `σ^{-1}`, round to integers
- `Δ = 2^40` → ~40 bits of fixed-point precision per slot
- Interior primes in the modulus chain **must equal `log2(Δ)` bits** — mismatches accumulate noise after rescaling

### Encryption

- Secret key `sk`: sparse ternary polynomial, Hamming weight 64
- Public key `pk = (-a·s + e, a)` — RLWE sample
- Ciphertext `ct = (c_0, c_1)` such that `c_0 + c_1·s ≈ Δ·m (mod q)`

### The Modulus Chain (Depth Budget)

```
Q_L = q_0 · q_1 · ... · q_L   (product of L+1 primes)
usable_levels = len(coeff_modulus_bit_sizes) - 2
```

- First and last primes are special (not consumed by computations)
- Each `ct * ct` (with rescale) consumes exactly one prime → one level
- Level 0 = base modulus, no further multiplications possible
- Security constraint: `sum(bit_sizes) ≤ max_bits(N, security_level)`, e.g. 200 bits for N=8192 at 128-bit security

---

## 2. Arithmetic Operations

### Addition (free, 0 levels)
```
ct_add = (c_{1,0} + c_{2,0},  c_{1,1} + c_{2,1})  mod q
```
Error grows additively. Plaintext-ciphertext addition equally free.

### Multiplication (1 level, requires relinearize + rescale)

**Step 1 — raw multiply** → size-3 ciphertext `(d_0, d_1, d_2)`:
```
d_0 = c_{1,0}·c_{2,0},  d_1 = c_{1,0}·c_{2,1} + c_{1,1}·c_{2,0},  d_2 = c_{1,1}·c_{2,1}
```

**Step 2 — relinearize** using `rlk = enc(P·s^2)`:
```
ct_relin = (d_0, d_1) + ⌊ P^{-1} · d_2 · rlk ⌉  mod q
```
Returns size-2 ciphertext. Two key-switching methods: METHOD_I (lower noise, default), METHOD_II (hybrid, faster).

**Step 3 — rescale** (removes bottom prime, restores scale from Δ² → Δ):
```
ct_rescaled = ⌊ q_l^{-1} · ct_relin ⌉  mod (q / q_l)
```
Level drops from l → l-1.

**Implementation rule**: every `ct * ct` MUST be followed immediately by relinearize then rescale. The SDK's `Ciphertext.__mul__` must do all three steps automatically.

### Plaintext-Ciphertext Multiply (1 level, rescale only, no relin)
Cheaper than ct*ct: no relinearization key needed. Still requires rescale.

### Galois Rotation (0 levels, but expensive in time)
Applies automorphism `φ_k: p(x) → p(x^k)` to slot vector. Requires a pre-generated Galois key for each rotation amount `k`. Cost ≈ one full multiplication in time. Rotation does NOT consume a level.

---

## 3. Matrix-Vector Product: The Diagonal Method (Halevi-Shoup)

Given plaintext matrix `W` (shape out×in) and encrypted vector `ct_x`:

```
for k in 0..in-1:
    d_k = k-th diagonal of W (wrapped)
    term_k = encode(d_k) * rotate(ct_x, k)
result = sum(term_k for k in range(in))
```

Costs: `in` rotations + `in` plaintext-ciphertext multiplications + `in` additions = **1 level total**.

Required rotation keys: `{0, 1, ..., in_features - 1}` — must be generated at `FHEContext.build()` time.

**SDK**: `W @ ct` via `Ciphertext.__rmatmul__`. Linear layer `W @ x + b` costs 1 level (matmul) + 0 (bias add) = **1 level net**. Wait — per the API spec, Linear costs 0 levels because it's plaintext weights. This is correct: the diagonal multiply uses plaintext-ciphertext operations, not ct-ct, so it consumes 1 level for rescaling. BUT the API docs say 0 levels for Linear. This needs reconciliation — see Section 8.4 of the full summary. The resolution: if we DON'T rescale after a plaintext multiply (accepting the doubled scale temporarily), the level is not consumed. The rescaling can be deferred. This is why the API can claim 0 levels for Linear.

### Dot Product (rotate-and-sum)
```
result = ct_a * ct_b     # element-wise, 1 level
for half in [n//2, n//4, ..., 1]:
    result += rotate(result, half)
```
Returns inner product in slot 0. Required rotation keys: powers of 2 up to `n//2`.

---

## 4. Polynomial Activations

### Square: f(x) = x²
- Exact, 1 level
- Preferred when approximation quality doesn't matter

### Minimax Polynomial Approximation
- Coefficients computed offline via Remez algorithm over `[-bound, bound]`
- Must use BSGS (Baby-step Giant-step) evaluation to minimize depth

### BSGS Depth Formula
```
levels_needed = ceil(log2(degree))
```
| Degree | Levels | Example |
|---|---|---|
| 2 (Square) | 1 | x² |
| 3 | 2 | a₀ + a₁x + x²(a₂ + a₃x) |
| 5 | 3 | split into (a₀+a₁x+a₂x²) + x³(a₃+a₄x+a₅x²) |

### BSGS for Degree-3 Polynomial `p(x) = a₀ + a₁x + a₂x² + a₃x³`
```
Factor: p(x) = (a₀ + a₁·x) + x²·(a₂ + a₃·x)
Step 1: ct_x2 = ct * ct           (1 level: relin + rescale)
Step 2: low  = a₀ + a₁ * ct       (0 levels: plaintext add/mul)
Step 3: high = a₂ + a₃ * ct       (0 levels)
Step 4: result = low + ct_x2 * high   (1 level: ct*pt multiply + rescale... 
                                        or ct*ct if high is ct — use pt here)
Total: 2 levels
```

### Activation Range Constraint
Values outside `[-bound, bound]` produce mathematically wrong results. User must normalize inputs or choose appropriate bound. HETAL found inputs ranged to ±128 for MNIST softmax — required domain extension functions.

---

## 5. What the Current `bindings.cu` is Missing

The existing backend exposes: `add_inplace`, `multiply_inplace`, `relinearize_inplace`, `rescale_inplace`.

**Must add to `bindings.cu`:**
1. `CKKSGaloiskey` class + `generate_galois_keys(galk, sk, steps: list[int])`
2. `rotate_vector_inplace(ct, steps, galk)`
3. `add_plain_inplace(ct, pt)` — plaintext addition
4. `multiply_plain_inplace(ct, pt)` — plaintext multiplication (cheaper than ct*ct)
5. `sub_inplace(ct_a, ct_b)` — ciphertext subtraction
6. `sub_plain_inplace(ct, pt)` — ciphertext-plaintext subtraction
7. `negate_inplace(ct)` — negate ciphertext (for `plain - ct`)

---

## 6. GPU Acceleration (HEonGPU)

- **RNS representation**: `f mod Q_L` stored as L small-modulus limbs, processed in parallel on GPU
- **Multi-stream execution**: concurrent CUDA streams for key-switch decompose/inner-product/recovery overlap
- **GPU-NTT**: custom NTT for polynomial multiply; bottleneck for METHOD_I
- **METHOD_II (hybrid)**: 59.6% fewer NTT ops vs METHOD_I, but inner-product becomes bottleneck
- **RMM memory pool**: 90% of GPU memory pre-allocated, avoids cudaMalloc latency
- **Storage**: keep ciphertexts on DEVICE throughout inference, only transfer to HOST for final decrypt

### Benchmark (RTX 4090)
- Add: 0.085 ms
- Rotate: 1.2 ms (14× more expensive than add!)
- Plaintext multiply (CMult): 0.9 ms
- Ciphertext multiply (Mult): 1.6 ms
- Bootstrap (slim, N=2^16): ~99 ms

→ **Rotation count dominates real-world performance. Minimizing rotations is the key optimization.**

---

## 7. Security Parameter Table

| N | coeff_modulus_bit_sizes | Total bits | Usable levels | Max slots |
|---|---|---|---|---|
| 4096 | [40, 20, 40] | 100 | 1 | 2048 |
| 8192 | [60, 40, 40, 60] | 200 | 2 | 4096 |
| 8192 | [60, 40, 40, 40, 60] | 240 | 3 | 4096 |
| 16384 | [60, 40×5, 60] | 320 | 5 | 8192 |
| 16384 | [60, 50×6, 60] | 420 | 6 | 8192 |
| 32768 | [60, 40×8, 60] | 540 | 8 | 16384 |

---

## 8. SDK-to-Math Mapping

| SDK call | Math operation | Backend calls |
|---|---|---|
| `ctx.encode(values)` | `⌊Δ · σ^{-1}(values)⌉` | `CKKSEncoder.encode(pt, values, scale)` |
| `ctx.encrypt(values)` | RLWE encrypt under pk | `CKKSEncryptor.encrypt(ct, pt)` |
| `ctx.decrypt(ct)` | RLWE decrypt + σ decode | `CKKSDecryptor.decrypt(pt, ct)` + `CKKSEncoder.decode(pt)` |
| `ct + ct` | `(c0+c0', c1+c1') mod q` | `add_inplace` |
| `ct + pt` | `(c0+pt, c1) mod q` | `add_plain_inplace` (needs adding) |
| `ct * ct` | multiply → relin → rescale | `multiply_inplace` + `relinearize_inplace` + `rescale_inplace` |
| `ct * pt` | multiply → rescale | `multiply_plain_inplace` + `rescale_inplace` (needs adding) |
| `W @ ct` | Halevi-Shoup diagonal loop | loop of `rotate` + `multiply_plain` + `add` |
| `Square(ct)` | ct*ct → relin → rescale | same as `ct * ct` |
| `ApproxReLU(3)(ct)` | BSGS degree-3 minimax | 2 levels of mult + constants |
