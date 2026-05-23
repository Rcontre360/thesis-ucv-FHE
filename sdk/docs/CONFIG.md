# CKKS Configuration Reference

Every knob on `FHEContext`, what it does, and the implications/performance
lessons learned while building and testing the SDK. CKKS is a tradeoff scheme:
no setting is free, and most of them pull against each other.

## Quick reference

| Setter | Default | What it controls |
|---|---|---|
| `set_poly_modulus_degree(n)` | — (required) | Ring dimension `N`. Slots, security headroom, memory, speed. |
| `set_coeff_modulus_bit_sizes([...])` | — (required) | The modulus chain: multiplicative levels + keyswitching primes. |
| `set_scale(s)` | — (required) | Fixed-point precision of encoded values. |
| `set_security_level(level)` | `SEC128` | Post-quantum security target, or `NONE` (insecure). |
| `set_bootstrapping_params(...)` | `(3, 3, 11)` | SLIM bootstrap circuit tuning. |
| `set_galois_key_storage(on_host)` | device | Whether rotation keys live in GPU VRAM or CPU RAM. |

Keyswitching method is **fixed to METHOD_II** (not user-configurable); see below.

---

## `poly_modulus_degree` — N

The degree of the polynomial ring. Must be a power of two. This is the master
parameter; everything else is sized relative to it.

- **Slots.** A ciphertext packs `N / 2` real values (the SIMD slot count). The
  SDK replicates the input cyclically across all slots for the diagonal matmul.
- **Security headroom.** Security depends on the ratio of `N` to the total
  modulus bits. A larger `N` allows a longer modulus chain at the same security
  level (see the cap table under *security_level*).
- **Memory.** Every polynomial is `N × (#primes)` 64-bit words, so memory grows
  **linearly with N**. Doubling `N` doubles VRAM use.
- **Speed.** NTTs are `O(N log N)`; larger `N` is slower per operation.

Typical values: `8192`, `16384`, `32768`, `65536`. Inference at 128-bit
security usually wants `16384`; secure bootstrapping needs `65536`.

## `coeff_modulus_bit_sizes` — the modulus chain

A list of prime bit-sizes. The CKKS ciphertext modulus is a product of these
primes, and **every multiplication consumes one** (it is dropped during the
rescale that follows).

**Layout the SDK uses:** all entries except the last are **Q primes** (the
usable chain); the last entry is the **P prime size**. So
`[60, 40, 40, 40, 40, 60]` means 5 Q primes and a 60-bit P.

- **Usable levels = (number of Q primes) − 1.** A fresh ciphertext from
  `[60,40,40,40,40,60]` has 4 usable multiplication levels.
- **Per-layer cost:** `Linear`/`Conv2D` = 1 level, `ReLU` = 2 levels. A
  `Conv → ReLU → Linear → ReLU → Linear` network costs 7 levels.
- **Prime-size convention:** the first prime is large (~60 bits) — it is the
  base prime and sets the precision floor. Middle primes should match the scale
  (~40 bits for `scale = 2^40`); each rescale strips one. The last (P) prime is
  large (~60 bits) for keyswitching accuracy.
- **P primes (keyswitching).** METHOD_II needs ≥2 P primes. The SDK auto-derives
  the count from the chain: `num_p = max(2, round(sum(Q) / (8 × P_size)))`,
  targeting `dnum ≈ 8`. A short chain gets 2; a long bootstrapping chain
  (~29 primes) gets ~3. The user supplies only one P size; the SDK replicates it.
- **Total bits are capped by security** (see below). The chain cannot be made
  arbitrarily long at a fixed `N`.

## `scale`

The fixed-point scaling factor applied when encoding. `2^40` means ~40 bits of
fractional precision. It must be consistent with the middle prime sizes: a
`2^40` scale wants ~40-bit interior primes, because each rescale divides the
ciphertext by roughly one prime and the scale must stay near it. Mismatched
scale and prime sizes cause precision loss or scale drift across levels.

## `security_level`

`SecurityLevel.SEC128` (default), `SEC192`, `SEC256`, or `NONE`.

The security level imposes a **cap on total modulus bits** (`sum(Q) + sum(P)`)
for the chosen `N`. Approximate 128-bit caps:

| N | ~max total modulus bits |
|---|---|
| 8192 | ~218 |
| 16384 | ~438 |
| 32768 | ~881 |
| 65536 | ~1750 |

Exceed the cap and `build()` raises. This is *the* reason deep circuits force
large `N`: more levels → more primes → more bits → larger `N` to stay secure.

`SecurityLevel.NONE` lifts the cap entirely. It is **insecure** and exists only
for demonstration/testing — e.g. running the bootstrap circuit (which needs a
~1500-bit chain) at small `N` on a modest GPU. Never use `NONE` for real data.

## Keyswitching method — METHOD_I vs METHOD_II

After a multiplication or rotation the ciphertext is briefly under a different
key; **keyswitching** restores it, using the relin/Galois keys. The SDK is
fixed to **METHOD_II** — it is not exposed as a setting — but the distinction
matters for understanding key memory.

- **METHOD_I (BV digit decomposition):** keyswitching keys scale as **O(L²)**
  in the chain length `L`. No P primes needed.
- **METHOD_II (hybrid keyswitching):** raises onto an extended modulus `Q·P`,
  decomposes into `dnum ≈ L / #P` groups. Keys scale as **O(L² / #P)** —
  several times smaller. Requires ≥2 P primes. Better noise growth, faster at
  large parameters.

For short chains the difference is minor. For a long bootstrapping chain
(`L ≈ 29`) METHOD_I's keys are enormous and overflow GPU memory; METHOD_II's
hybrid keys fit. This is why the SDK uses METHOD_II everywhere.

## `set_bootstrapping_params(ctos_piece, stoc_piece, taylor_number)`

Tuning for the SLIM bootstrap circuit. Defaults `(3, 3, 11)` are a middle
ground; bootstrapping itself is automatic (see *Bootstrapping* below).

- **`ctos_piece` / `stoc_piece`** (range 2–5): the CoeffToSlot / SlotToCoeff
  homomorphic DFTs are factored into this many sub-matrices. More pieces →
  sparser factors, fewer rotations each, but **more circuit depth**. Fewer
  pieces → denser factors, more rotations, less depth.
- **`taylor_number`** (range 6–15): degree of the polynomial approximating the
  EvalMod sine. Higher → more accurate refresh, but more depth. `11` gives
  ~10⁻³ precision, ample for inference.

These do **not** depend on the network — only on `N` and target precision.

## `set_galois_key_storage(on_host)`

Where rotation (Galois) keys live.

- **Device (default):** all keys resident in GPU VRAM. Fast — no transfer per
  rotation — but all keys occupy VRAM at once.
- **Host (`on_host=True`):** keys live in CPU RAM; each is streamed to the GPU
  only while a rotation uses it. Far less VRAM (one key resident, not all), at
  the cost of a host→device copy per rotation. Bootstrapping needs ~30 keys;
  host storage is what makes them fit on a small GPU.

---

## The level-budget model

Without bootstrapping, a network is bounded by the chain: `sum(layer depths)`
must not exceed the usable levels. This is the dominant limit on network depth.

With bootstrapping, `Sequential.compile(context)` measures the budget, and if
the network overflows it, enables SLIM bootstrapping and statically schedules
refreshes between layers — depth becomes effectively unbounded. If the network
fits, no bootstrapping is set up at all (it is skipped, saving the keys and
setup cost).

## Bootstrapping — what it costs

Bootstrapping refreshes an exhausted ciphertext back to near-full levels. The
SLIM circuit (SlotToCoeff → ModRaise → CoeffToSlot → EvalMod) is itself
**~25–30 levels deep**, so:

- The modulus chain must be long enough for the circuit *plus* the network
  between refreshes — a ~29-prime chain is typical.
- A ~1500-bit chain that long only stays 128-bit secure at **N=65536**, which
  needs roughly **16 GB+ of VRAM**.
- At smaller `N` the circuit only fits with `SecurityLevel.NONE` (insecure).

Measured precision of one SLIM bootstrap (insecure long chain): ~19 bits at
N=4096, ~18 at N=8192, ~16 at N=16384.

## Hardware sizing — measured on a 4 GB GPU

- Non-bootstrapping inference: a secure `N=32768` context builds fine and gives
  ~15 usable levels.
- Bootstrapping fits at **N ≤ 16384** with `SecurityLevel.NONE` + host-stored
  keys. `N=32768` overflows GPU memory; secure `N=65536` is far out of reach.
- `nvidia-smi` is misleading here: the GPU memory manager (RMM) pre-reserves a
  fixed pool, so `memory.used` reports the pool envelope, not real consumption.
  Actual usage scales linearly with `N`; trust the out-of-memory error, not the
  reported figure.

## Recipes

```python
# Standard secure inference (shallow network, no bootstrapping)
FHEContext().set_poly_modulus_degree(16384) \
    .set_coeff_modulus_bit_sizes([60, 40, 40, 40, 40, 60]) \
    .set_scale(2**40).build()

# Deep network with bootstrapping — demo only, INSECURE params
FHEContext().set_poly_modulus_degree(16384) \
    .set_coeff_modulus_bit_sizes([60] + [50] * 28 + [60]) \
    .set_scale(2**50) \
    .set_security_level(SecurityLevel.NONE) \
    .set_galois_key_storage(on_host=True).build()
```
