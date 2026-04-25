#!/usr/bin/env python3
"""Plaintext baseline benchmark — PyTorch SquareNet, CPU.

Network: Input(64) -> Linear(64->64) -> Square (x^2) -> Linear(64->10)

Provides the timing and accuracy baseline that all FHE scripts are compared against.
Run this first to understand the overhead each FHE library adds.

Run from sdk/ root:
    python examples/bench-dev/plaintext_bench.py
"""
import os, sys, time
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
from network import (
    IN_FEATURES, HIDDEN, N_CLASSES, FHE_BATCH,
    load_dataset, train_squarenet, plaintext_accuracy, plaintext_logits,
)

N_WARMUP = 5
N_TIMED  = 50

# ---------------------------------------------------------------------------
# 1. Dataset + training
# ---------------------------------------------------------------------------

print("=" * 55)
print("Plaintext Baseline Benchmark (PyTorch)")
print(f"Network : Input({IN_FEATURES}) -> Linear({IN_FEATURES}->{HIDDEN}) "
      f"-> Square -> Linear({HIDDEN}->{N_CLASSES})")
print("=" * 55)

X_train, X_test, y_train, y_test, X_fhe, y_fhe = load_dataset()
print(f"\n[1] Training SquareNet (30 epochs)...")
t0    = time.perf_counter()
model = train_squarenet(X_train, y_train)
train_time = time.perf_counter() - t0
print(f"    done in {train_time:.1f}s")
print(f"    test set accuracy   : {plaintext_accuracy(model, X_test, y_test):.4f}")
print(f"    FHE-batch accuracy  : {plaintext_accuracy(model, X_fhe, y_fhe):.4f}")

# ---------------------------------------------------------------------------
# 2. Per-sample inference timing (single sample at a time — FHE comparison)
# ---------------------------------------------------------------------------

print(f"\n[2] Per-sample inference timing  "
      f"({N_WARMUP} warmup + {N_TIMED} timed runs each)...")

sample_times = []
for i, x in enumerate(X_fhe):
    x_t = torch.tensor([x], dtype=torch.float32)
    model.eval()
    with torch.no_grad():
        for _ in range(N_WARMUP):
            _ = model(x_t)
    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(N_TIMED):
            out = model(x_t)
    elapsed_ms = (time.perf_counter() - t0) / N_TIMED * 1000
    sample_times.append(elapsed_ms)
    pred = int(out.argmax(1))
    print(f"    [{i+1:2d}/{FHE_BATCH}] true={y_fhe[i]} pred={pred} | {elapsed_ms:.4f} ms")

avg_single_ms = float(np.mean(sample_times))
print(f"\n    avg per-sample : {avg_single_ms:.4f} ms")

# ---------------------------------------------------------------------------
# 3. Batched inference timing (full FHE_BATCH at once)
# ---------------------------------------------------------------------------

print(f"\n[3] Batched inference timing  ({FHE_BATCH} samples, "
      f"{N_WARMUP} warmup + {N_TIMED} timed runs)...")
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
print(f"    {FHE_BATCH}-sample batch : {batch_ms:.4f} ms total  "
      f"({batch_ms/FHE_BATCH:.4f} ms/sample)")

# ---------------------------------------------------------------------------
# 4. Results
# ---------------------------------------------------------------------------

print(f"\n{'=' * 55}")
print("Results")
print(f"{'=' * 55}")
print(f"  Test set accuracy       : {plaintext_accuracy(model, X_test, y_test):.4f}")
print(f"  FHE-batch accuracy      : {plaintext_accuracy(model, X_fhe, y_fhe):.4f}")
print(f"  Avg per-sample (single) : {avg_single_ms:.4f} ms")
print(f"  Avg per-sample (batch)  : {batch_ms/FHE_BATCH:.4f} ms")
