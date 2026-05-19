"""Regression benchmark — shallow MLP, our CKKS SDK.

Trains a depth-7 ReLU MLP on California Housing (PyTorch, plaintext), ports
the frozen weights into the SDK, calibrates per-neuron activation ranges on
the training data (with a margin), runs encrypted inference using the
polynomial ReLU (Cheon f_n composite), and reports latency / accuracy.

Because the f_n composite is deep (~70 multiplicative levels per ReLU), this
benchmark uses INSECURE parameters (`SecurityLevel.NONE`) and SLIM
bootstrapping. Secure-parameter runs would need a >= 16 GB GPU.
"""

import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
from sklearn.metrics import r2_score

from helpers.data import load_california_housing
from helpers.training import get_trained_model

from api import FHEContext
from api.layers.linear import Linear
from api.functions.relu import ReLU
from api.sequential import Sequential
from core.enums import SecurityLevel

# f_2 (degree 5) is the deepest f_n whose per-composition unit (4 levels) fits
# one SLIM bootstrap cycle on this chain (after_boot 7, stoc 3: 7-4 >= 3).
# 12 compositions; one bootstrap fires per composition.
RELU_DEGREES = (5,) * 12
N_TEST = 5

# Long chain so SLIM bootstrapping has room. Total Q+P bits ~1640 — insecure
# at N=16384, the whole point of SecurityLevel.NONE.
COEFF_MODULUS = [60] + [50] * 28 + [60]


def to_sdk_model(torch_model: torch.nn.Module) -> Sequential:
    layers = []
    for module in torch_model:
        if isinstance(module, torch.nn.Linear):
            w = module.weight.detach().numpy()
            b = module.bias.detach().numpy()
            layers.append(Linear(w.shape[1], w.shape[0], w, b))
        else:
            layers.append(ReLU(degrees=RELU_DEGREES))
    return Sequential(layers)


def gpu_mem_mib() -> int:
    out = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"]
    )
    return int(out.decode().split("\n")[0])


print("loading data + training plaintext model...")
data = load_california_housing()
torch_model = get_trained_model("shallow", data)

print("building FHE context (INSECURE long chain + host-stored keys)...")
# N=8192: Galois keys scale with the ring degree, so half the degree of
# N=16384 means half-size bootstrapping keys — robust headroom in the ~3.4 GB
# RMM pool. 4096 slots still comfortably hold the widest layer (128).
ctx = (
    FHEContext()
    .set_poly_modulus_degree(8192)
    .set_coeff_modulus_bit_sizes(COEFF_MODULUS)
    .set_scale(2**50)
    .set_security_level(SecurityLevel.NONE)
    .set_galois_key_storage(on_host=True)
    .build()
)
print(f"usable levels: {ctx._usable_levels()}")

model = to_sdk_model(torch_model)
print(f"network mult_depth: {sum(l.mult_depth() for l in model._layers)}")

print("compiling (calibrating + setting up SLIM bootstrapping)...")
t0 = time.perf_counter()
model.compile(ctx, calibration_data=data.x_train)
print(f"  compile: {time.perf_counter() - t0:.1f}s")
print(f"  bootstrapping enabled: {ctx._bootstrapping_ready} "
      f"(refreshes fire lazily, including mid-ReLU)")

x_test = data.x_test[:N_TEST]
y_test = data.y_test[:N_TEST]
with torch.no_grad():
    plain = torch_model(torch.tensor(x_test, dtype=torch.float32)).numpy().reshape(-1)

print(f"running encrypted inference on {N_TEST} samples...")
enc = np.empty(N_TEST)
t0 = time.perf_counter()
for i, x in enumerate(x_test):
    enc[i] = model(model.input(ctx, x.tolist())).decrypt()[0]
elapsed = time.perf_counter() - t0

print()
print("=== shallow MLP (real ReLU train, f_n composite encrypted) ===")
print(f"{'plaintext R2':<26}: {r2_score(y_test, plain):.4f}")
print(f"{'encrypted R2':<26}: {r2_score(y_test, enc):.4f}")
print(f"{'enc vs plain max |err|':<26}: {np.abs(enc - plain).max():.6f}")
print(f"{'enc vs plain mean |err|':<26}: {np.abs(enc - plain).mean():.6f}")
print(f"{'latency':<26}: {elapsed / N_TEST:.1f} s/sample")
print(f"{'GPU memory (RMM pool)':<26}: {gpu_mem_mib()} MiB")
