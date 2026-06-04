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

from fhe_ml import BootstrapConfig, FHEConfig, FHEContext, SecurityLevel, Sequential
from fhe_ml.layers import Linear

DEPTH = 30

config = FHEConfig(
    log_n=14,
    coeff_modulus_bit_sizes=[60] + [50] * 28 + [60],
    log_scale=50,
    security_level=SecurityLevel.NONE,
    galois_keys_on_host=True,
    bootstrap=BootstrapConfig(),
)
ctx = FHEContext(config).build()

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
