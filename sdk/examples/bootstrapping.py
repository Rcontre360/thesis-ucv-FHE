"""Encrypted inference deeper than the level budget — automatic bootstrapping.

A stack of identity Linear layers performs more plaintext multiplications than
the CKKS modulus chain has levels. `Sequential.compile()` detects this and sets
up SLIM bootstrapping; the layers then refresh the ciphertext lazily during
inference as levels run low. The output still equals the input because every
layer is the identity — so any drift is purely CKKS/bootstrapping noise, which
makes correctness easy to read.

INSECURE PARAMETERS: SecurityLevel.NONE lifts the modulus-bit cap so the long
chain the bootstrap circuit needs fits at N=16384 on a 4 GB GPU. This is a
correctness demo only — never use NONE for real data. A secure run needs
N=65536 and a >=16 GB GPU.

Run: from the sdk/ dir, `/usr/bin/python3.12 examples/bootstrapping.py`.
"""

from api import FHEContext
from api.layers.linear import Linear
from api.sequential import Sequential
from core.enums import SecurityLevel

DEPTH = 30  # identity Linear layers — deeper than the ~28-level fresh budget

# Long modulus chain (29 Q primes) so the SLIM bootstrap circuit fits.
ctx = (
    FHEContext()
    .set_poly_modulus_degree(16384)
    .set_coeff_modulus_bit_sizes([60] + [50] * 28 + [60])
    .set_scale(2**50)
    .set_security_level(SecurityLevel.NONE)   # INSECURE — demo only
    .set_galois_key_storage(on_host=True)     # stream keys from CPU RAM: fits 4 GB
    .build()
)

identity = [
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.0],
    [0.0, 0.0, 0.0, 1.0],
]
model = Sequential([Linear(4, 4, identity) for _ in range(DEPTH)])

print(f"network depth : {DEPTH} levels")
print(f"fresh budget  : {ctx._usable_levels()} levels  -> network overflows it")

model.compile(ctx)
print(f"bootstrapping enabled: {ctx._bootstrapping_ready}  "
      f"(refreshes fire lazily during inference)")

data = [0.1, 0.2, 0.3, 0.4]
result = model(model.input(ctx, data)).decrypt()

print(f"\ninput  : {data}")
print(f"output : {[round(v, 4) for v in result]}")
print(f"max error vs input: {max(abs(result[i] - data[i]) for i in range(4)):.5f}")
