#!/usr/bin/env python3
"""fhe_sdk GPU-CKKS benchmark.

Network: Input(64) -> Linear(64->64) -> Square (x^2) -> Linear(64->10)

Run from sdk/ root:
    python examples/classifier/sdk_bench.py
"""
import time
import numpy as np
from sklearn.metrics import accuracy_score

from network import (
    IN_FEATURES, HIDDEN, N_CLASSES, FHE_BATCH,
    load_dataset, train_squarenet, plaintext_accuracy,
    plaintext_logits, extract_weights,
)
from api import FHEContext
from api.layers.linear import Linear
from api.sequential import Sequential

POLY_MOD_DEGREE     = 16384
COEFF_MOD_BIT_SIZES = [60, 40, 40, 40, 60]
SCALE               = 2 ** 40

X_train, X_test, y_train, y_test, X_fhe, y_fhe = load_dataset()
model = train_squarenet(X_train, y_train)
W1, b1, W2, b2 = extract_weights(model)

t0  = time.perf_counter()
ctx = (
    FHEContext()
    .set_poly_modulus_degree(POLY_MOD_DEGREE)
    .set_coeff_modulus_bit_sizes(COEFF_MOD_BIT_SIZES)
    .set_scale(SCALE)
    .build()
)
keygen_ms = (time.perf_counter() - t0) * 1000


class Square:
    def __call__(self, x):
        return x * x


fhe_model = Sequential([
    Linear(IN_FEATURES, HIDDEN, W1, bias=b1),
    Square(),
    Linear(HIDDEN, N_CLASSES, W2, bias=b2),
])


def forward(x_plain):
    t_enc = time.perf_counter()
    enc   = ctx.encrypt(x_plain)
    t_enc = time.perf_counter() - t_enc

    t_inf = time.perf_counter()
    enc_out = fhe_model(enc)
    t_inf = time.perf_counter() - t_inf

    t_dec = time.perf_counter()
    out   = enc_out.decrypt()
    t_dec = time.perf_counter() - t_dec

    return out[:N_CLASSES], {'enc': t_enc, 'inf': t_inf, 'dec': t_dec}


preds, timings = [], []
for x in X_fhe:
    logits, t = forward(x.tolist())
    preds.append(int(np.argmax(logits)))
    timings.append(t)

avg_enc = np.mean([t['enc'] for t in timings]) * 1000
avg_inf = np.mean([t['inf'] for t in timings]) * 1000
avg_dec = np.mean([t['dec'] for t in timings]) * 1000
avg_tot = avg_enc + avg_inf + avg_dec

pt_out  = plaintext_logits(model, [X_fhe[0].tolist()])
fhe_out = np.array(forward(X_fhe[0].tolist())[0])
max_err = np.abs(pt_out - fhe_out).max()

print(f"{'=' * 55}")
print("Results — fhe_sdk CKKS (GPU)")
print(f"{'=' * 55}")
print(f"  FHE accuracy            : {accuracy_score(y_fhe, preds):.4f}")
print(f"  Plaintext accuracy      : {plaintext_accuracy(model, X_fhe, y_fhe):.4f}")
print(f"  Key generation          : {keygen_ms:8.1f} ms")
print(f"  Avg encrypt             : {avg_enc:8.1f} ms")
print(f"  Avg inference           : {avg_inf:8.1f} ms")
print(f"  Avg decrypt             : {avg_dec:8.1f} ms")
print(f"  Avg total               : {avg_tot:8.1f} ms")
print(f"  Correctness max |err|   : {max_err:.2e}")
