#!/usr/bin/env python3
"""Plaintext baseline benchmark — PyTorch SquareNet, CPU.

Network: Input(64) -> Linear(64->64) -> Square (x^2) -> Linear(64->10)

Run from sdk/ root:
    python examples/classifier/plaintext_bench.py
"""
import time
import numpy as np
import torch

from network import (
    IN_FEATURES, HIDDEN, N_CLASSES, FHE_BATCH,
    load_dataset, train_squarenet, plaintext_accuracy,
)

N_WARMUP = 5
N_TIMED  = 50

X_train, X_test, y_train, y_test, X_fhe, y_fhe = load_dataset()
model = train_squarenet(X_train, y_train)

# Per-sample inference timing
sample_times = []
for x in X_fhe:
    x_t = torch.tensor([x], dtype=torch.float32)
    model.eval()
    with torch.no_grad():
        for _ in range(N_WARMUP):
            _ = model(x_t)
    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(N_TIMED):
            out = model(x_t)
    sample_times.append((time.perf_counter() - t0) / N_TIMED * 1000)

avg_single_ms = float(np.mean(sample_times))

# Batched inference timing
X_fhe_t = torch.tensor(X_fhe, dtype=torch.float32)
model.eval()
with torch.no_grad():
    for _ in range(N_WARMUP):
        _ = model(X_fhe_t)
t0 = time.perf_counter()
with torch.no_grad():
    for _ in range(N_TIMED):
        _ = model(X_fhe_t)
batch_ms = (time.perf_counter() - t0) / N_TIMED * 1000

print(f"{'=' * 55}")
print("Results — Plaintext Baseline (PyTorch)")
print(f"{'=' * 55}")
print(f"  Test set accuracy       : {plaintext_accuracy(model, X_test, y_test):.4f}")
print(f"  FHE-batch accuracy      : {plaintext_accuracy(model, X_fhe, y_fhe):.4f}")
print(f"  Avg per-sample (single) : {avg_single_ms:.4f} ms")
print(f"  Avg per-sample (batch)  : {batch_ms/FHE_BATCH:.4f} ms")
