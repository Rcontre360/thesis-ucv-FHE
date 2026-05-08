#!/usr/bin/env python3
"""Compare SDK diagonal matmul against numpy for a small matrix.

Run from sdk/:
    python examples/matmul_compare.py
"""
import numpy as np

from api import FHEContext
from api.tensor import PlaintextTensor

N = 30
np.random.seed(42)
W = np.random.randn(N, N).astype(float)
x = np.random.randn(N).astype(float)

expected = W @ x

ctx = (
    FHEContext()
    .set_poly_modulus_degree(16384)
    .set_coeff_modulus_bit_sizes([60, 40, 40, 60])
    .set_scale(2 ** 40)
    .build()
)

enc = ctx.encrypt(x.tolist())
result = enc.matmul(PlaintextTensor(W.tolist()))
got = np.array(result.decrypt()[:N])

print(f"{'i':>4}  {'expected':>12}  {'got':>12}  {'abs_err':>12}")
print("-" * 46)
for i in range(N):
    print(f"{i:>4}  {expected[i]:>12.6f}  {got[i]:>12.6f}  {abs(expected[i]-got[i]):>12.2e}")

print()
print(f"max |err| = {np.abs(expected - got).max():.4e}")
print(f"mean|err| = {np.abs(expected - got).mean():.4e}")
