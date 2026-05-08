"""Simple two-layer neural network over encrypted data.

Architecture:  input(4) -> Linear(4->8) -> ReLU -> Linear(8->2) -> output(2)

All arithmetic runs homomorphically on GPU via CKKS.
Weights and biases are plaintext; only the input is encrypted.
"""

from api import FHEContext
from api.layers.linear import Linear
from api.functions.activations import ReLU
from api.sequential import Sequential

# ---------------------------------------------------------------------------
# Weights and biases (normally loaded from a trained model)
# ---------------------------------------------------------------------------

W1 = [
    [ 0.5, -0.3,  0.8,  0.1],
    [-0.2,  0.7, -0.4,  0.6],
    [ 0.9, -0.1,  0.3, -0.5],
    [-0.6,  0.4,  0.2,  0.8],
    [ 0.1, -0.9,  0.5, -0.2],
    [ 0.7,  0.3, -0.6,  0.4],
    [-0.3,  0.8,  0.1, -0.7],
    [ 0.4, -0.5,  0.9,  0.2],
]
b1 = [0.1, -0.1, 0.05, -0.05, 0.2, -0.2, 0.15, -0.15]

W2 = [
    [ 0.3, -0.2,  0.5, -0.1,  0.4, -0.3,  0.2, -0.4],
    [-0.1,  0.6, -0.3,  0.5, -0.2,  0.4, -0.5,  0.1],
]
b2 = [0.05, -0.05]

# ---------------------------------------------------------------------------
# Input (would normally be the sample to classify / regress)
# ---------------------------------------------------------------------------

plaintext_input = [0.6, -0.4, 0.8, -0.2]

# ---------------------------------------------------------------------------
# Build FHE context and model
# ---------------------------------------------------------------------------

print("Building FHE context...")
ctx = FHEContext.default()

model = Sequential([
    Linear(4, 8, W1, bias=b1),
    ReLU(),
    Linear(8, 2, W2, bias=b2),
])

# ---------------------------------------------------------------------------
# Encrypt input and run inference
# ---------------------------------------------------------------------------

print(f"Input:  {plaintext_input}")

encrypted_input = ctx.encrypt(plaintext_input)
encrypted_output = model(encrypted_input)
result = encrypted_output.decrypt()

print(f"Output: {[round(v, 4) for v in result]}")

# ---------------------------------------------------------------------------
# Plain reference (no encryption) for sanity check
# ---------------------------------------------------------------------------

def plain_linear(x, W, b):
    return [
        sum(W[i][j] * x[j] for j in range(len(x))) + b[i]
        for i in range(len(W))
    ]

def plain_relu(x):
    return [0.125 * v**2 + 0.5 * v + 0.375 for v in x]

h = plain_relu(plain_linear(plaintext_input, W1, b1))
expected = plain_linear(h, W2, b2)

print(f"Expected (plaintext): {[round(v, 4) for v in expected]}")
print(f"Max error: {max(abs(result[i] - expected[i]) for i in range(2)):.6f}")
