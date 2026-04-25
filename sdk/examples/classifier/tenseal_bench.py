#!/usr/bin/env python3
"""TenSEAL CKKS benchmark — cpu-only reference.

Network: Input(64) -> Linear(64->64) -> Square (x^2) -> Linear(64->10)

Run from sdk/ root:
    python examples/bench-dev/tenseal_bench.py
"""
import os, sys, time
import numpy as np
import tenseal as ts

sys.path.insert(0, os.path.dirname(__file__))
from network import (
    IN_FEATURES, HIDDEN, N_CLASSES, FHE_BATCH,
    load_dataset, train_squarenet, plaintext_accuracy,
    plaintext_logits, extract_weights,
)

POLY_MOD_DEGREE     = 16384
COEFF_MOD_BIT_SIZES = [60, 40, 40, 40, 60]   # 3 usable levels: mm + square + mm
SCALE               = 2 ** 40

# ---------------------------------------------------------------------------
# 1. Dataset + plaintext model
# ---------------------------------------------------------------------------

print("=" * 55)
print("TenSEAL CKKS Benchmark")
print(f"Network : Input({IN_FEATURES}) -> Linear({IN_FEATURES}->{HIDDEN}) "
      f"-> Square -> Linear({HIDDEN}->{N_CLASSES})")
print("=" * 55)

X_train, X_test, y_train, y_test, X_fhe, y_fhe = load_dataset()
print(f"\n[1] Training PyTorch SquareNet ({30} epochs)...")
t0    = time.perf_counter()
model = train_squarenet(X_train, y_train)
print(f"    done in {time.perf_counter()-t0:.1f}s  "
      f"| test acc: {plaintext_accuracy(model, X_test, y_test):.4f}")

W1, b1, W2, b2 = extract_weights(model)
# TenSEAL mm(W) expects W in [in_features, out_features] column-major layout
W1_mm = np.array(W1).T.tolist()   # [64, 64]
W2_mm = np.array(W2).T.tolist()   # [64, 10]

# ---------------------------------------------------------------------------
# 2. TenSEAL context
# ---------------------------------------------------------------------------

print(f"\n[2] Building TenSEAL CKKS context...")
t0 = time.perf_counter()
ctx = ts.context(ts.SCHEME_TYPE.CKKS,
                 poly_modulus_degree=POLY_MOD_DEGREE,
                 coeff_mod_bit_sizes=COEFF_MOD_BIT_SIZES)
ctx.generate_galois_keys()
ctx.global_scale = SCALE
keygen_ms = (time.perf_counter() - t0) * 1000

print(f"    poly_modulus_degree : {POLY_MOD_DEGREE}")
print(f"    coeff_mod_bit_sizes : {COEFF_MOD_BIT_SIZES}")
print(f"    usable CKKS levels  : {len(COEFF_MOD_BIT_SIZES) - 2}")
print(f"    key generation      : {keygen_ms:.1f} ms")

# ---------------------------------------------------------------------------
# 3. Encrypted inference
# ---------------------------------------------------------------------------

def forward(x_plain):
    t_enc = time.perf_counter()
    enc   = ts.ckks_vector(ctx, x_plain)
    t_enc = time.perf_counter() - t_enc

    t_inf = time.perf_counter()
    enc = enc.mm(W1_mm) + b1    # Linear(64->64)
    enc = enc * enc              # Square
    enc = enc.mm(W2_mm) + b2    # Linear(64->10)
    t_inf = time.perf_counter() - t_inf

    t_dec = time.perf_counter()
    out   = enc.decrypt()
    t_dec = time.perf_counter() - t_dec

    return out[:N_CLASSES], {'enc': t_enc, 'inf': t_inf, 'dec': t_dec}


print(f"\n[3] Encrypted inference on {FHE_BATCH} samples...")
preds, timings = [], []
for i, x in enumerate(X_fhe):
    logits, t = forward(x.tolist())
    preds.append(int(np.argmax(logits)))
    timings.append(t)
    tot = sum(t.values()) * 1000
    print(f"    [{i+1:2d}/{FHE_BATCH}] true={y_fhe[i]} pred={preds[-1]} | "
          f"enc={t['enc']*1000:6.1f} ms  "
          f"inf={t['inf']*1000:6.1f} ms  "
          f"dec={t['dec']*1000:6.1f} ms  "
          f"tot={tot:6.1f} ms")

# ---------------------------------------------------------------------------
# 4. Results
# ---------------------------------------------------------------------------

from sklearn.metrics import accuracy_score
avg_enc = np.mean([t['enc'] for t in timings]) * 1000
avg_inf = np.mean([t['inf'] for t in timings]) * 1000
avg_dec = np.mean([t['dec'] for t in timings]) * 1000
avg_tot = avg_enc + avg_inf + avg_dec

# Correctness: compare decrypted logits vs plaintext on sample 0
pt_out  = plaintext_logits(model, [X_fhe[0].tolist()])
fhe_out = np.array(forward(X_fhe[0].tolist())[0])
max_err = np.abs(pt_out - fhe_out).max()

print(f"\n{'=' * 55}")
print("Results")
print(f"{'=' * 55}")
print(f"  FHE accuracy            : {accuracy_score(y_fhe, preds):.4f}")
print(f"  Plaintext accuracy      : {plaintext_accuracy(model, X_fhe, y_fhe):.4f}")
print(f"  Avg encrypt             : {avg_enc:8.1f} ms")
print(f"  Avg inference           : {avg_inf:8.1f} ms")
print(f"  Avg decrypt             : {avg_dec:8.1f} ms")
print(f"  Avg total               : {avg_tot:8.1f} ms")
print(f"  Correctness max |err|   : {max_err:.2e}")
