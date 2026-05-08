#!/usr/bin/env python3
"""Alternating mul / add / rotate stress test: numpy vs SDK.

Each round:  result = result * weights[r]
             result = result + biases[r]
             result = rotate(result, shifts[r])

Run from sdk/:
    python examples/ops_compare.py
"""
import numpy as np
from api import FHEContext

ROUNDS = 7
N      = 8
np.random.seed(0)

x       = np.random.randn(N)
weights = [np.random.randn(N) for _ in range(ROUNDS)]
biases  = [np.random.randn(N) for _ in range(ROUNDS)]
shifts  = [1, 2, 4, 1, 2, 4, 1]   # must be powers of 2 (those are the only Galois keys generated)

# --- Plaintext reference ---
# SDK rotate(k): left-rotation over slot_count=8192 slots.
# The vector is zero-padded: slots N..8191 are 0.
# After rotating by k, the last k positions receive those zeros, not the
# wrapped values that np.roll would produce.
def ckks_rotate(arr, k):
    result = np.zeros(N)
    result[:N - k] = arr[k:N]
    return result

ref = x.copy()
refs = []
for r in range(ROUNDS):
    ref = ref * weights[r]
    ref = ref + biases[r]
    ref = ckks_rotate(ref, shifts[r])
    refs.append(ref.copy())

# --- SDK encrypted ---
# 7 plain-multiplies need 7 usable levels → 8 Q primes.
# n=16384, 7 middle 40-bit primes + first 60-bit + P 60-bit = 400 bits ≤ 438 max.
ctx = (
    FHEContext()
    .set_poly_modulus_degree(16384)
    .set_coeff_modulus_bit_sizes([60] + [40] * 7 + [60])
    .set_scale(2 ** 40)
    .build()
)

enc = ctx.encrypt(x.tolist())

for r in range(ROUNDS):
    enc = enc * weights[r].tolist()
    enc = enc + biases[r].tolist()
    enc = enc.rotate(shifts[r])
    got = np.array(enc.decrypt()[:N])
    err = np.abs(refs[r] - got)

    print(f"\n=== round {r+1}  (rotate {shifts[r]}) ===")
    print(f"{'i':>3}  {'numpy':>12}  {'sdk':>12}  {'abs_err':>10}")
    print("-" * 42)
    for i in range(N):
        print(f"{i:>3}  {refs[r][i]:>12.6f}  {got[i]:>12.6f}  {err[i]:>10.2e}")
    print(f"     max|err|={err.max():.2e}  mean|err|={err.mean():.2e}")
