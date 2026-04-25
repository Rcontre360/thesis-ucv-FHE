#!/usr/bin/env python3
"""Concrete-ML TFHE benchmark.

Network: Input(64) -> Linear(64->64) -> Square (x^2) -> Linear(64->10)

Uses compile_torch_model (not NeuralNetClassifier) so that the GPU path
works correctly — the NeuralNetClassifier compile path has a known bug
where Configuration(use_gpu=True) is not propagated.

Run from sdk/ root:
    python examples/classifier/concrete_bench.py
"""
import os, time
import subprocess
import numpy as np
import torch
import concrete.compiler
from concrete.compiler import check_gpu_available
from concrete.ml.torch.compile import compile_torch_model

from network import (
    IN_FEATURES, HIDDEN, N_CLASSES, FHE_BATCH,
    SquareNet, load_dataset, train_squarenet, plaintext_accuracy,
)

N_BITS = 6   # quantisation bit-width — accuracy vs speed tradeoff


def gpu_snapshot(label: str):
    """Print a one-line GPU snapshot via nvidia-smi."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi",
             "--query-gpu=utilization.gpu,utilization.memory,memory.used,memory.total",
             "--format=csv,noheader,nounits"],
            text=True,
        ).strip()
        print(f"  [GPU {label}] util={out}")
    except Exception as e:
        print(f"  [GPU {label}] nvidia-smi failed: {e}")


# ---------------------------------------------------------------------------
# 0. Header
# ---------------------------------------------------------------------------

use_gpu = check_gpu_available()
device  = "cuda" if use_gpu else "cpu"

print("=" * 55)
print("Concrete-ML TFHE Benchmark  (compile_torch_model)")
print(f"  GPU enabled (runtime): {concrete.compiler.check_gpu_enabled()}")
print(f"  GPU available (hw)   : {use_gpu}")
print(f"  compile device       : {device}")
gpu_snapshot("startup")
print(f"Network : Input({IN_FEATURES}) -> Linear({IN_FEATURES}->{HIDDEN}) "
      f"-> Square -> Linear({HIDDEN}->{N_CLASSES})")
print(f"n_bits  : {N_BITS}")
print("=" * 55)

# ---------------------------------------------------------------------------
# 1. Dataset
# ---------------------------------------------------------------------------

X_train, X_test, y_train, y_test, X_fhe, y_fhe = load_dataset()

# ---------------------------------------------------------------------------
# 2. Train PyTorch model
# ---------------------------------------------------------------------------

print(f"\n[1] Training SquareNet (30 epochs)...")
t0         = time.perf_counter()
model      = train_squarenet(X_train, y_train)
train_time = time.perf_counter() - t0
pt_acc     = plaintext_accuracy(model, X_test, y_test)
print(f"    done in {train_time:.1f}s  |  plaintext accuracy: {pt_acc:.4f}")

# ---------------------------------------------------------------------------
# 3. FHE compilation
#    compile_torch_model correctly propagates device='cuda' to the backend,
#    unlike NeuralNetClassifier which has a known _compiled_for_cuda bug.
# ---------------------------------------------------------------------------

print(f"\n[2] Compiling to FHE circuit (device={device!r})...")
gpu_snapshot("pre-compile")
t0           = time.perf_counter()
q_module     = compile_torch_model(
    model,
    X_train,
    rounding_threshold_bits=N_BITS,
    p_error=0.05,
    device=device,
)
compile_time = time.perf_counter() - t0
print(f"    compile time: {compile_time:.1f}s")
print(f"    _compiled_for_cuda: {q_module._compiled_for_cuda}")
gpu_snapshot("post-compile")

# ---------------------------------------------------------------------------
# 4. Key generation
# ---------------------------------------------------------------------------

print(f"\n[3] Key generation...")
gpu_snapshot("pre-keygen")
t0          = time.perf_counter()
q_module.fhe_circuit.keygen()
keygen_time = time.perf_counter() - t0
print(f"    keygen time: {keygen_time:.1f}s")
gpu_snapshot("post-keygen")

# ---------------------------------------------------------------------------
# 5. Simulated FHE accuracy
# ---------------------------------------------------------------------------

sim_preds = q_module.forward(X_fhe.astype(np.float32), fhe="simulate")
sim_acc   = (sim_preds.argmax(1) == y_fhe).mean()
print(f"\n    simulated FHE acc (batch={FHE_BATCH}): {sim_acc:.4f}")

# ---------------------------------------------------------------------------
# 6. Encrypted inference
# ---------------------------------------------------------------------------

print(f"\n[4] Encrypted inference on {FHE_BATCH} samples...")
gpu_snapshot("pre-execute")
t0         = time.perf_counter()
fhe_logits = q_module.forward(X_fhe.astype(np.float32), fhe="execute")
total_infer = time.perf_counter() - t0
per_sample_ms = total_infer / FHE_BATCH * 1000
gpu_snapshot("post-execute")

fhe_preds = fhe_logits.argmax(1)
for i, (pred, true) in enumerate(zip(fhe_preds, y_fhe)):
    print(f"    [{i+1:2d}/{FHE_BATCH}] true={true} pred={pred}")

# ---------------------------------------------------------------------------
# 7. Results
# ---------------------------------------------------------------------------

fhe_acc = (fhe_preds == y_fhe).mean()

print(f"\n{'=' * 55}")
print("Results")
print(f"{'=' * 55}")
print(f"  Plaintext accuracy      : {pt_acc:.4f}")
print(f"  Simulated FHE accuracy  : {sim_acc:.4f}")
print(f"  FHE accuracy (batch)    : {fhe_acc:.4f}")
print(f"  Compile time            : {compile_time:8.1f} s")
print(f"  Key generation          : {keygen_time:8.1f} s")
print(f"  Total inference time    : {total_infer:8.1f} s  ({FHE_BATCH} samples)")
print(f"  Per-sample inference    : {per_sample_ms:8.1f} ms")
