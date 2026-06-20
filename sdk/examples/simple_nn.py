"""Simple two-layer neural network over encrypted data.

Architecture:  input(4) -> Linear(4->8) -> ReLU -> Linear(8->2) -> output(2)

Client/server split: the server holds the model and an evaluator context (no
secret key); the client owns the secret key, encrypts the input, and decrypts
the result. Weights/biases are plaintext.
"""

import numpy as np

from fhe_ml import Client, FHEContext, Sequential
from fhe_ml.layers import Linear, ReLU

# ---------------------------------------------------------------------------
# Weights and biases (normally loaded from a trained model)
# ---------------------------------------------------------------------------

W1 = [
    [0.5, -0.3, 0.8, 0.1],
    [-0.2, 0.7, -0.4, 0.6],
    [0.9, -0.1, 0.3, -0.5],
    [-0.6, 0.4, 0.2, 0.8],
    [0.1, -0.9, 0.5, -0.2],
    [0.7, 0.3, -0.6, 0.4],
    [-0.3, 0.8, 0.1, -0.7],
    [0.4, -0.5, 0.9, 0.2],
]
b1 = [0.1, -0.1, 0.05, -0.05, 0.2, -0.2, 0.15, -0.15]

W2 = [
    [0.3, -0.2, 0.5, -0.1, 0.4, -0.3, 0.2, -0.4],
    [-0.1, 0.6, -0.3, 0.5, -0.2, 0.4, -0.5, 0.1],
]
b2 = [0.05, -0.05]

# ---------------------------------------------------------------------------
# Input (would normally be the sample to classify / regress)
# ---------------------------------------------------------------------------

plaintext_input = [0.6, -0.4, 0.8, -0.2]

# ---------------------------------------------------------------------------
# Build FHE context and model
# ---------------------------------------------------------------------------

print("Building server context + compiling model...")
model = Sequential(
    [
        Linear(4, 8, W1, bias=b1),
        ReLU(),
        Linear(8, 2, W2, bias=b2),
    ]
)
# The SDK derives the entire CKKS config (prime chain, P primes, log_n,
# bootstrapping) from just the scale and the ReLU polynomial degrees.
config = model.generate_config(scale=40, relu_degrees=(3,))
ctx = FHEContext(config)  # server: no secret key
model.compile(ctx, np.array([plaintext_input], dtype=np.float32))

# ---------------------------------------------------------------------------
# Client: owns the secret key, encrypts the input, decrypts the result
# ---------------------------------------------------------------------------

client = Client(ctx)  # generates keys for the compiled model
ctx.set_client_params(client.key_params())  # server loads the public eval keys

print(f"Input:  {plaintext_input}")

encrypted_output = model(client.encrypt(plaintext_input))
result = client.decrypt(encrypted_output)[:2]

print(f"Output: {[round(v, 4) for v in result]}")

# ---------------------------------------------------------------------------
# Plain reference: the SDK's own cleartext path (same polynomial ReLU)
# ---------------------------------------------------------------------------

expected = model.forward_plain(np.asarray(plaintext_input))[:2]
print(f"Expected (plaintext): {[round(float(v), 4) for v in expected]}")
print(f"Max error: {max(abs(result[i] - expected[i]) for i in range(2)):.6f}")
