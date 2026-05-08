#!/usr/bin/env python3
"""Concrete-ML TFHE benchmark.

Network: Input(64) -> Linear(64->64) -> Square (x^2) -> Linear(64->10)

Run from sdk/ root:
    python examples/classifier/concrete_bench.py
"""
import time
import numpy as np
import torch
from concrete.compiler import check_gpu_available
from concrete.ml.torch.compile import compile_torch_model

from network import (
    IN_FEATURES, HIDDEN, N_CLASSES, FHE_BATCH,
    SquareNet, load_dataset, train_squarenet, plaintext_accuracy,
)

N_BITS = 6

X_train, X_test, y_train, y_test, X_fhe, y_fhe = load_dataset()
model = train_squarenet(X_train, y_train)
pt_acc = plaintext_accuracy(model, X_test, y_test)

device = "cuda" if check_gpu_available() else "cpu"

t0 = time.perf_counter()
q_module = compile_torch_model(
    model,
    X_train,
    rounding_threshold_bits=N_BITS,
    p_error=0.05,
    device=device,
)
compile_time = time.perf_counter() - t0

t0 = time.perf_counter()
q_module.fhe_circuit.keygen()
keygen_time = time.perf_counter() - t0

sim_out   = q_module.forward(X_fhe.astype(np.float32), fhe="simulate")
sim_acc   = (sim_out.argmax(1) == y_fhe).mean()

t0 = time.perf_counter()
fhe_logits = q_module.forward(X_fhe.astype(np.float32), fhe="execute")
total_infer = time.perf_counter() - t0

fhe_preds = fhe_logits.argmax(1)
fhe_acc   = (fhe_preds == y_fhe).mean()

print(f"{'=' * 55}")
print("Results — Concrete-ML TFHE")
print(f"{'=' * 55}")
print(f"  Device                  : {device}")
print(f"  Plaintext accuracy      : {pt_acc:.4f}")
print(f"  Simulated FHE accuracy  : {sim_acc:.4f}")
print(f"  FHE accuracy            : {fhe_acc:.4f}")
print(f"  Compile time            : {compile_time:8.1f} s")
print(f"  Key generation          : {keygen_time:8.1f} s")
print(f"  Total inference time    : {total_infer:8.1f} s  ({FHE_BATCH} samples)")
print(f"  Per-sample inference    : {total_infer/FHE_BATCH*1000:8.1f} ms")
