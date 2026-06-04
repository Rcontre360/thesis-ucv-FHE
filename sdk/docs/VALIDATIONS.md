# FHEConfig validations

All validations run on `FHEConfig` / `BootstrapConfig` construction **and on every subsequent attribute assignment**, with rollback to the previous valid value on rejection. They sit at the config-object boundary, before `FHEContext.build()` ever calls into the backend, so a misconfiguration fails immediately at the call site.

The backend used is [HEonGPU](https://github.com/Alisah-Ozcan/HEonGPU/) (RNS-CKKS on CUDA).

Every check below documents:
- **What** it enforces.
- **Why** the constraint exists.
- **Source** — the upstream rule it mirrors, or the empirical justification if we added it ourselves.

---

## 1. `log_n` — supported polynomial degrees

`log_n ∈ {12, 13, 14, 15, 16}` (i.e. `N ∈ {4096, 8192, 16384, 32768, 65536}`).

**Why.** CKKS works in the ring `Z[X]/(X^N + 1)` with `N` a power of 2 — this is what makes the NTT-based arithmetic both efficient and correct. HEonGPU additionally caps the upper end at `2^16` for build-time and memory reasons.

**Source.** HEonGPU `set_poly_modulus_degree` rejects non-powers-of-2 and values outside `[MIN_POLY_DEGREE=4096, MAX_POLY_DEGREE=65536]`:
- [`defines.h:14-15`](https://github.com/Alisah-Ozcan/HEonGPU/blob/main/src/include/heongpu/kernel/defines.h#L14-L15) defines the constants.
- [`ckks/context.cu:36-46`](https://github.com/Alisah-Ozcan/HEonGPU/blob/main/src/lib/host/ckks/context.cu#L36-L46) does the rejection.

We store `log_n` (the exponent) instead of `N` itself, which makes the power-of-2 constraint unrepresentable by construction.

---

## 2. `coeff_modulus_bit_sizes` shape

Must be a list with `len ≥ 2`. The last entry is the size of the P (special / keyswitching) primes; everything before is the Q chain.

**Why.** Our `FHEContext.build()` always uses METHOD_II keyswitching (`num_p ≥ 2`), and the Q chain needs at least one prime to be useful. A zero-Q chain or missing P entry is meaningless.

**Source.** Mirrors HEonGPU's METHOD_II requirement: [`ckks/context.cu:75-84`](https://github.com/Alisah-Ozcan/HEonGPU/blob/main/src/lib/host/ckks/context.cu#L75-L84) throws if `log_P_bases_bit_sizes.size() < 2` for METHOD_II.

---

## 3. `coeff_modulus_bit_sizes` per-entry bit range

Each entry must be an int in `[30, 60]`.

**Why.** HEonGPU's `generate_primes` rejects any prime size outside this range at build time. The lower bound (30) is what makes the NTT-friendly prime generator work reliably for all supported `N`; below it the generator throws `"invalid modulus bit size"`. The upper bound (60) is the largest size that fits in a `Data64` (uint64) with room for modular arithmetic without overflow.

**Source.** [`defines.h:18-19`](https://github.com/Alisah-Ozcan/HEonGPU/blob/main/src/include/heongpu/kernel/defines.h#L18-L19):
```cpp
#define MAX_USER_DEFINED_MOD_BIT_COUNT 60
#define MIN_USER_DEFINED_MOD_BIT_COUNT 30
```
Enforced in [`util.cu:251-257`](https://github.com/Alisah-Ozcan/HEonGPU/blob/main/src/lib/util/util.cu#L251-L257) inside `generate_primes`.

**Empirical confirmation.** `examples/precision_floor.py` sweeps prime sizes 10–40 at `log_n=13`. Every size below 30 raises `"invalid modulus bit size"` at build; at 30 bits the round-trip relative error after three `ct * plain` multiplications is ~1.5e-6 (well under 1%). Confirms HEonGPU's floor and that nothing smaller is reachable through the API.

---

## 4. `log_scale` positivity

`log_scale ≥ 1`.

**Why.** Scale = `2^log_scale` is the multiplicative factor applied during encoding. Zero or negative scale produces no encoded magnitude — the plaintext is unrecoverable.

**Source.** Our own check. HEonGPU's encoder accepts any positive double for scale and produces silent garbage outside meaningful ranges; this minimal check catches the most degenerate input.

**Note — no upper bound for scale.** We deliberately do not enforce `log_scale ≤ max(Q bits)` or `≤ min(Q bits)`. The CKKS-theoretic upper bound is "scale must fit inside the rescaling primes", but the exact rule depends on which Q primes are used as scale-carrying vs. as rescale targets, and HEonGPU doesn't enforce one either. Users can pick scales that lose precision; this is a deliberate trade-off between flexibility and safety. May be revisited if real bugs surface.

---

## 5. `bootstrap` type

`bootstrap` is `Optional[BootstrapConfig]`. None means "this configuration does not need bootstrapping"; setting a `BootstrapConfig` opts in.

**Why.** Most short-chain configurations have no business bootstrapping — bootstrapping only kicks in when `Sequential.compile()` detects a network deeper than the level budget. Making it opt-in lets the default `FHEConfig` (a short, lightweight chain) be valid without forcing users to disable a feature they never asked for.

**Source.** Our own design decision. HEonGPU has no equivalent — `generate_bootstrapping_params` happily configures bootstrap parameters even for chains too short to use them; the failure surfaces later as a runtime error inside the bootstrap operation. Making bootstrap opt-in at the config layer makes the intent explicit and enables the chain-depth check (validation #8 below).

---

## 6. Security cap

If `security_level != NONE`, the total of all coefficient modulus bits (Q + P) must be ≤ the HE-Std cap for the chosen `(security_level, log_n)`.

**Why.** Going over the cap means the lattice problem underlying CKKS is solvable in fewer operations than the security level claims. Above the cap, the parameters are insecure — the ciphertexts can be broken with effort below the advertised bit-security.

**Source.** Caps copied directly from HEonGPU [`secstdparams.h`](https://github.com/Alisah-Ozcan/HEonGPU/blob/main/src/include/heongpu/util/secstdparams.h), which mirror the [lattice-estimator](https://github.com/malb/lattice-estimator) tool's estimates for ternary secret, error std-dev 3.2.

HEonGPU enforces the same cap in [`ckks/context.cu:108-133`](https://github.com/Alisah-Ozcan/HEonGPU/blob/main/src/lib/host/ckks/context.cu#L108-L133); replicating it here means the error fires at the config object instead of inside the backend.

---

## 7. `coefficient_validator` — Q-chunk vs. P-total

For every consecutive chunk of `num_p` Q primes (where `num_p` is the count of P primes our `build()` replicates), the sum of the Q chunk's bit sizes must be ≤ the total P bits.

**Why.** This is METHOD_II keyswitching's correctness requirement: the P primes must be large enough (collectively) to absorb the noise from any window of Q primes during a keyswitch. A chunk exceeding total P bits produces incorrect keyswitching.

**Source.** Direct Python port of HEonGPU's [`coefficient_validator`](https://github.com/Alisah-Ozcan/HEonGPU/blob/main/src/lib/util/util.cu#L11-L55). HEonGPU calls it from [`ckks/context.cu:86-90`](https://github.com/Alisah-Ozcan/HEonGPU/blob/main/src/lib/host/ckks/context.cu#L86-L90):
```cpp
if (!coefficient_validator(log_Q_bases_bit_sizes, log_P_bases_bit_sizes))
{
    throw std::logic_error("P should be bigger than Q pairs!");
}
```

---

## 8. Bootstrap chain depth (when `bootstrap is not None`)

`Q_size ≥ ctos_piece + stoc_piece + taylor_number + 9`, where `Q_size = len(coeff_modulus_bit_sizes) - 1`.

**Why.** A SLIM bootstrap raises a ciphertext at level `1 + stoc_piece`, runs CoeffToSlot (costs `ctos_piece` levels), EvalMod (costs `taylor_number + 8` levels for cleanup), and SlotToCoeff. The output sits at level `Q_size - ctos_piece - taylor_number - 8`. For that output to be usable as the input to **another** bootstrap (which our `Sequential.compile` pipeline assumes — bootstraps fire repeatedly during inference of a deep network), we need:

```
Q_size - ctos_piece - taylor_number - 8 ≥ 1 + stoc_piece
  ⟺  Q_size ≥ ctos_piece + stoc_piece + taylor_number + 9
```

**Source.** Derived from HEonGPU's bootstrap-level setup in [`ckks/operator.cu`](https://github.com/Alisah-Ozcan/HEonGPU/blob/main/src/lib/host/ckks/operator.cu) (the `generate_bootstrapping_params` body for SLIM_BOOTSTRAPPING sets `StoC_level = 1 + StoC_piece`, `CtoS_level = Q_size`, and the bootstrap pipeline consumes `ctos_piece + taylor_number + 8` levels).

HEonGPU does **not** enforce this preemptively. Its check fires only at the moment the bootstrap operation runs, as `Ciphertexts leveled should be at max!` in [`operator.cu:6332-6337`](https://github.com/Alisah-Ozcan/HEonGPU/blob/main/src/lib/host/ckks/operator.cu#L6332-L6337):
```cpp
int current_decomp_count = Q_size_ - input1.depth_;
if (current_decomp_count != (1 + StoC_piece_))
{
    throw std::logic_error("Ciphertexts leveled should be at max!");
}
```
By the time that throws, you've already paid for key generation, model compilation and possibly several inference steps. Validating at the config object turns this into an immediate, actionable error.

---

## Validations HEonGPU has that we deliberately don't

- **Order-of-call enforcement.** HEonGPU throws if `set_poly_modulus_degree` is called after `set_coeff_modulus_bit_sizes`, etc. Our `FHEConfig` is a plain dataclass — order of attribute writes doesn't matter, only the final state. No equivalent check needed.

- **Internal `KEYSWITCHING_METHOD_I` constraint** (`P chain ≤ 1`). We always pick METHOD_II via our `num_p = max(2, ...)` formula in `build()`, so the constraint is unreachable from user input.

- **Bootstrap input-level check** at the moment of `slim_bootstrapping` invocation. We validate at config time via the chain-depth rule above; the backend check is a defense-in-depth backstop, but our error surfaces earlier and with a more actionable message.

## Validations we have that HEonGPU does not

- **`log_scale` positivity** (validation #4).
- **`bootstrap` opt-in semantics** + chain-depth check at config time (validation #5 + #8 — HEonGPU has neither).
- **`BootstrapConfig` field ranges** (`ctos_piece`, `stoc_piece` ∈ `[2, 5]`; `taylor_number` ∈ `[6, 15]`). HEonGPU does not validate these — out-of-range values may produce broken bootstrap circuits or runtime errors. Our ranges come from the documented usable range in HEonGPU's BootstrappingConfig usage.
