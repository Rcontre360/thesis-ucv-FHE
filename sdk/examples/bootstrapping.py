"""Encrypted inference deeper than the level budget — automatic bootstrapping.

A stack of identity Linear layers performs more plaintext multiplications than
the CKKS modulus chain has levels. `generate_config` sizes the chain and enables
SLIM bootstrapping when the circuit overflows the budget; `compile()` wires it
up, and the layers refresh the ciphertext lazily during inference. The output
still equals the input because every layer is the identity — so any drift is
purely CKKS/bootstrapping noise, which makes correctness easy to read.

`generate_config` uses the secure default (SEC128, N=2^16), so bootstrapping
here needs a GPU with enough memory for log_n=16 (a 4 GB card is not enough; use
>= 16 GB).

Run: from the sdk/ dir, `/usr/bin/python3.12 examples/bootstrapping.py`.
"""

import numpy as np

from fhe_ml import Client, FHEContext, Sequential
from fhe_ml.layers import Linear

DEPTH = 40  # deeper than the auto-sized budget at N=2^16, so it must bootstrap

identity = [
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.0],
    [0.0, 0.0, 0.0, 1.0],
]
model = Sequential([Linear(4, 4, identity) for _ in range(DEPTH)])

# Pure-linear network (no ReLU), so relu_degrees is an unused placeholder.
config = model.generate_config(scale=50, relu_degrees=(7,))
config.galois_keys_on_host = True  # keep the bootstrap galois keys off-device
ctx = FHEContext(config)

print(f"network depth : {DEPTH} levels")
print(f"fresh budget  : {ctx._usable_levels()} levels  -> network overflows it")

data = [0.1, 0.2, 0.3, 0.4]
model.compile(ctx, np.array([data]))
client = Client(ctx)  # owns the secret key; makes the eval keys
ctx.set_client_params(client.key_params())  # server loads relin + galois keys
print(
    f"bootstrapping enabled: {ctx._bootstrapping_ready}  "
    f"(refreshes fire lazily during inference)"
)

result = client.decrypt(model(model.input(client, data)))

print(f"\ninput  : {data}")
print(f"output : {[round(v, 4) for v in result]}")
print(f"max error vs input: {max(abs(result[i] - data[i]) for i in range(4)):.5f}")
