# Matrix–vector multiplication in `fhe_sdk`

This document explains how `EncryptedVector.matmul` computes
`y = W @ x` when `x` is a CKKS-encrypted vector and `W` is a plaintext
matrix. It assumes you already understand CKKS slot-packing, rotations,
and plaintext multiplications.

The algorithm is the **rectangular Halevi–Shoup diagonal method with
cyclic-wrap diagonals on a replicated input**. It is the same algorithm
TenSEAL uses (`diagonal_ct_vector_matmul`); the only difference is that
we feed the ciphertext into the rotation-side of the loop instead of
the plaintext-side (mathematically equivalent, fewer Galois keys).

---

## 1. The slot-pattern convention

A CKKS ciphertext carries `slot_count = N/2` independent slots. We use a
**replicated** convention: an encrypted vector of logical size `n` lives
in *every* slot of the ciphertext via cyclic repetition.

```
logical x = [x0, x1, x2, x3]                 (n = 4)

ciphertext slots  :   0    1    2    3    4    5    6    7    8   …
encrypted values  :  x0   x1   x2   x3   x0   x1   x2   x3   x0   …
                     └──── period 4 ────┘└──── period 4 ────┘
```

Properties this convention gives us:

- Rotations preserve the replication: `rotate(x, k)[i] = x[(i+k) mod n]`.
- The first `n` slots are *the* answer — `decrypt` reads slots `[0, n)`.
- Plaintext arithmetic (`+ scalar`, `+ list[float]`, `+ Plaintext`)
  encodes its right-hand side replicated to `slot_count` too, so slot
  patterns always match without explicit bookkeeping.

There is no "tile period < `slot_count`" state to track. No
zero-padding region. Every slot is meaningful.

---

## 2. Restating matrix-vector multiply with cyclic indices

For `W ∈ ℝ^{out × in}` and `x ∈ ℝ^{in}`, the matrix-vector product is

```
y[c] = Σ_{r=0}^{in-1} W[c][r] · x[r]                for c ∈ [0, out)
```

The Halevi–Shoup rewrite re-indexes the sum so that the row index of
`W` and the index of `x` are **rotated together** by the same offset.
Define `M = Wᵀ` of shape `(in, out)`; then with `i ≡ r − c` taken
modulo `in`,

```
y[c] = Σ_{i=0}^{in-1} M[(i+c) mod in][c]  ·  x[(i+c) mod in]
                       └── matrix entry ──┘    └─ rotated x ─┘
                       this is diag_i[c]
```

So for each `i ∈ [0, in)` we get a length-vector

```
diag_i[k] = M[(i+k) mod in][k mod out]
          = W[k mod out][(i+k) mod in]                       ← our convention
```

and

```
y[k mod out]  =  Σ_i  diag_i[k]  ·  rot(x, i)[k]
```

The crucial observation: this identity holds for **every** slot `k` (not
just `k < out`), because both sides of the equation depend on `k mod
out` only. That is what produces the **replicated output**: the result
ciphertext satisfies `result[k] = y[k mod out]` for all `k`.

### Why both row and column wrap

`(i+k) mod in` wraps the row of `M`; `k mod out` wraps the column.
This is what makes the algorithm work for arbitrary `(in, out)` shapes
— square, wide, tall, coprime, non-coprime — without any zero-padding.
For square `in == out` the walk degenerates to the textbook square
diagonals; for non-square it is the generalisation due to TenSEAL.

---

## 3. The algorithm — pictorial

For `n_rows = in_features` iterations:

```
        ┌───────────────┐
input:  │   enc_x       │   replicated: x[k mod n_rows] in slot k
        └───────┬───────┘
                │
       ┌────────┴─────────────────────────────────────────────┐
       │   loop: for i in [0, n_rows):                         │
       │                                                       │
       │      diag_i[k] = W[k mod n_cols][(i+k) mod n_rows]    │
       │      ───────────────────────────────────────          │
       │      length n_rows·n_cols, replicated to slot_count   │
       │                                                       │
       │      term  = rotated  ⊙  encode(diag_i)               │  (multiply_plain)
       │      term  = rescale(term)                            │
       │      result += term                                   │  (homomorphic add)
       │                                                       │
       │      rotated = rotate(rotated, 1)                     │  (CKKS rotate)
       └───────┬───────────────────────────────────────────────┘
               │
        ┌──────┴────────┐
output: │   result      │   replicated: y[k mod n_cols] in slot k
        └───────────────┘
```

Three things happen per iteration:

1. Build `diag_i` as a length-`n_rows·n_cols` plain vector (a CPU walk
   through the matrix entries).
2. Multiply the **rotated input** by `diag_i` slot-wise.
3. Roll `rotated` forward by one rotation for the next iteration.

The rotation always has step 1 — no need for arbitrary-shift Galois
keys, just the rotate-by-1 key (and powers of two for `_sum_slots`).

---

## 4. A worked 3×2 example

Take `W = [[a, b], [c, d], [e, f]]` (so `out_features = 3`,
`in_features = 2`) and `x = [x0, x1]`. Expected result:

```
y = W @ x = [ a·x0 + b·x1,
              c·x0 + d·x1,
              e·x0 + f·x1 ]                           (size 3)
```

`n_rows = in = 2`, `n_cols = out = 3`. Two iterations, each producing a
length-6 diagonal that is then replicated to `slot_count`.

### Diagonal 0 (`i = 0`)

```
diag_0[k] = W[k mod 3][(0+k) mod 2]   for k = 0, 1, …, 5

k=0 →  W[0][0] = a
k=1 →  W[1][1] = d
k=2 →  W[2][0] = e
k=3 →  W[0][1] = b
k=4 →  W[1][0] = c
k=5 →  W[2][1] = f

diag_0 = [a, d, e, b, c, f]      then replicated: [a,d,e,b,c,f,a,d,e,b,c,f, …]
```

### Diagonal 1 (`i = 1`)

```
diag_1[k] = W[k mod 3][(1+k) mod 2]   for k = 0, 1, …, 5

k=0 →  W[0][1] = b
k=1 →  W[1][0] = c
k=2 →  W[2][1] = f
k=3 →  W[0][0] = a
k=4 →  W[1][1] = d
k=5 →  W[2][0] = e

diag_1 = [b, c, f, a, d, e]      replicated similarly
```

### Trace of slot 0 (= `y[0]`)

The input `enc_x` has slot pattern `[x0, x1, x0, x1, x0, x1, …]`.

Iteration `i = 0` (rotated = enc_x):

```
slot 0 of (rotated ⊙ diag_0)  =  x0 · a
```

Iteration `i = 1` (rotated = rotate(enc_x, 1) = `[x1, x0, x1, x0, …]`):

```
slot 0 of (rotated ⊙ diag_1)  =  x1 · b
```

Sum: `slot 0 of result = a·x0 + b·x1 = y[0]`. ✓

### Trace of slot 1 (= `y[1]`)

Iteration `i = 0`:

```
slot 1 of (enc_x ⊙ diag_0)  =  enc_x[1] · diag_0[1]  =  x1 · d
```

Iteration `i = 1` (rotated has `x0` at slot 1):

```
slot 1 of (rotated ⊙ diag_1)  =  x0 · c
```

Sum: `slot 1 of result = d·x1 + c·x0 = y[1]`. ✓

### Trace of slot 2 (= `y[2]`)

Iteration `i = 0`: `enc_x[2] · diag_0[2] = x0 · e`.
Iteration `i = 1`: `rotated[2] · diag_1[2] = x1 · f`.
Sum: `e·x0 + f·x1 = y[2]`. ✓

### Trace of slot 3 (should = `y[3 mod 3] = y[0]`)

Iteration `i = 0`: `enc_x[3] · diag_0[3] = x1 · b`.
Iteration `i = 1`: `rotated[3] · diag_1[3] = x0 · a`.
Sum: `b·x1 + a·x0 = y[0]`. ✓ — the output is **replicated** as
promised.

---

## 5. Behaviour by matrix shape

The same algorithm runs for every `(in, out)` pair. What changes is the
structure of the diagonals.

```
┌──────────────────┬───────────────┬─────────────────────────┬──────────────────┐
│ Case             │ Diagonal      │ Each diag covers        │ # of iterations  │
│                  │ "period"      │                         │ (= n_rows)       │
├──────────────────┼───────────────┼─────────────────────────┼──────────────────┤
│ Square (in==out) │   in          │ one cyclic diagonal of  │   in             │
│                  │               │ W, the textbook H–S     │                  │
│                  │               │ object — `in` cells     │                  │
├──────────────────┼───────────────┼─────────────────────────┼──────────────────┤
│ Coprime          │   in·out      │ ALL `in·out` cells of   │   in             │
│ (gcd=1)          │   (= lcm)     │ W, in a CRT-permuted    │                  │
│                  │               │ order                   │                  │
├──────────────────┼───────────────┼─────────────────────────┼──────────────────┤
│ Non-coprime      │   lcm(in,out) │ `lcm(in,out)` cells,    │   in             │
│ non-square       │               │ each visited            │                  │
│                  │               │ (in·out)/lcm times      │                  │
└──────────────────┴───────────────┴─────────────────────────┴──────────────────┘
```

Concrete chapter-3 layers:

| Layer                | shape    | gcd | diag period | "case"          |
|----------------------|----------|-----|-------------|-----------------|
| `Linear(64, 64)`     | 64×64    | 64  |  64         | square          |
| `Linear(784, 784)`   | 784×784  | 784 | 784         | square          |
| `Linear(784, 64)`    | 784×64   | 16  | 3 136       | non-coprime     |
| `Linear(64, 10)`     | 64×10    |  2  | 320         | non-coprime     |

For square layers each iteration's diagonal carries only `in` distinct
values (one cyclic diagonal of W). For non-square layers the diagonal
walks through more cells, but the **number of iterations stays the
same**: `n_rows = in_features`.

---

## 6. Cost summary

Per matmul, ignoring the all-zero-diagonal skip optimisation:

```
n_rows  multiply_plain  + n_rows  rescale  + (n_rows − 1)  CKKS rotations
                                              └ all rotate-by-1, single Galois key ┘

multiplicative depth consumed: 1
```

The all-zero skip in `EncryptedVector.matmul` (`if not all(v == 0.0
for v in diag)`) elides the multiply/rescale on dense-zero diagonals
but still performs the rotate-by-1 advance, since `rotated` must reach
the right offset for the next non-zero diagonal.

For a typical dense neural-network layer the loop is dominated by
`n_rows` CKKS rotations, each ~one NTT + one key-switch.

---

## 7. What the algorithm depends on at the API surface

`EncryptedVector.matmul(matrix)` requires:

- `matrix.ndim == 2`.
- `matrix.shape == (out_features, in_features)`.
- `self._n_values == in_features`.
- `self` was produced by `FHEContext.encrypt(...)` (so it carries the
  replicated slot pattern), or by an upstream operation that preserves
  replication. **Every operation in this SDK preserves the replicated
  slot pattern**: `matmul`, `+ scalar`, `+ list`, `* scalar`, `* list`,
  `Plaintext` arithmetic, ciphertext-ciphertext add/sub/multiply,
  rotation. So once you encrypt, you stay in the convention.

`EncryptedVector.matmul` returns an `EncryptedVector` of size
`out_features`, with the slot pattern `result[k] = y[k mod
out_features]` everywhere in the ciphertext.

---

## 8. Failure modes

The algorithm computes the correct answer for every shape that fits
the slot count. A handful of edge cases produce errors or undefined
behaviour:

- **`in_features != self._n_values`** — raises `ValueError("Matrix
  columns ... != vector size ...")`. Caught at the top of `matmul`.
- **`in_features > slot_count`** or **`in_features · out_features >
  slot_count`** without truncation handling — would silently truncate
  the diagonal vector. Not encountered in chapter 3 (`max in_features
  = 784`, `slot_count ≥ 4096`), but a future limitation if very wide
  layers are added.
- **All-zero W** — raises `ValueError("All matrix diagonals are
  zero")`. Should never happen with trained weights.
- **Slots near `slot_count − 1`** — replicated rotation produces a
  glitchy band of size `< n_rows` near the end of the slot vector.
  No downstream operation reads those slots, so this is invisible
  in practice. Documented in `MATMUL_ISSUES.md` (L5).

---

## 9. Relation to other variants

The algorithm in this SDK is one specific point in the design space of
plaintext-matrix × ciphertext-vector multiplication.

```
              ┌───────────────────────────────────────────────────────────┐
              │ Halevi–Shoup family                                       │
              │   ─────────────                                           │
              │                                                           │
              │   Square zero-pad to (s × s):   our previous SDK version  │
              │      pros: textbook, simple math                          │
              │      cons: L1, L2, L3 of MATMUL_ISSUES.md;                │
              │            extra rotations on tall matrices               │
              │                                                           │
              │   Rectangular cyclic-wrap:      ★ this SDK ★              │
              │                                  TenSEAL (post-2020)      │
              │      pros: native non-square, replicated I/O,             │
              │            no period bookkeeping                          │
              │      cost: O(n_rows) rotations                            │
              │                                                           │
              │   BSGS (baby-step / giant-step):  future optimisation     │
              │      pros: O(√n_rows) rotations                           │
              │      cost: more plaintexts, more code                     │
              │                                                           │
              │   Specialised conv (im2col + tree-sum):                   │
              │      pros: O(log kernel_size) rotations for conv          │
              │      cost: distinct encoding mode and op (`conv2d`)       │
              │            See TenSEAL `conv2d_im2col`. Out of scope      │
              │            until CNN benchmarks demand it.                │
              └───────────────────────────────────────────────────────────┘
```

This SDK ships the **rectangular cyclic-wrap** variant. Convolutions
are currently expressed as sparse Toeplitz matmuls — correct but
expensive; a dedicated conv2d primitive is anticipated future work.

---

## 10. Code pointer

The implementation lives in `src/api/ciphertext.py:matmul`. The two
helpers it relies on are:

- `FHEContext.encode` (replicates `values` to `slot_count`).
- `EncryptedVector._encode_and_align` (encodes a length-`n` value list
  at the matching depth as the ciphertext).

Tests covering the algorithm's contract are in
`tests/test_ciphertext.py::TestEncryptedVectorMatmul` — identity, wide
projection, tall expansion, chained narrow-after-wide, and shape /
type validation.
