"""Regression benchmark — deep MLP, our CKKS SDK.

Mirrors `sdk_shallow.py` but uses the deep network (`build_deep`, 21 ReLU layers
~ 43 layers total). The composite ReLU is depth-heavy and the deep stack
amplifies it: ~250+ bootstraps per sample. Insecure parameters required.

If this is too slow or fails to train, that's expected — California Housing
doesn't need a 43-layer MLP, and deep ReLU MLPs vanish without skip
connections. The point is to exercise repeated mid-layer bootstrapping at
scale, not to chase accuracy.
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

RELU_DEGREES = (5,) * 12   # same as sdk_shallow.py — only choice that fits the cycle
N_TEST = 3
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


print("loading data + training plaintext deep model...")
data = load_california_housing()
torch_model = get_trained_model("deep", data)

print("building FHE context (INSECURE long chain + host-stored keys)...")
ctx = (
    FHEContext()
    .set_poly_modulus_degree(8192)
    .set_coeff_modulus_bit_sizes(COEFF_MODULUS)
    .set_scale(2**50)
    .set_security_level(SecurityLevel.NONE)
    .set_galois_key_storage(on_host=True)
    .build()
)

model = to_sdk_model(torch_model)
print(f"network: {len(model._layers)} layers, mult_depth {sum(l.mult_depth() for l in model._layers)}")

print("compiling (calibrating + setting up SLIM bootstrapping)...")
t0 = time.perf_counter()
model.compile(ctx, calibration_data=data.x_train)
print(f"  compile: {time.perf_counter() - t0:.1f}s")
print(f"  bootstrapping enabled: {ctx._bootstrapping_ready}")

x_test = data.x_test[:N_TEST]
y_test = data.y_test[:N_TEST]
with torch.no_grad():
    plain = torch_model(torch.tensor(x_test, dtype=torch.float32)).numpy().reshape(-1)

print(f"running encrypted inference on {N_TEST} samples...")
enc = np.empty(N_TEST)
t0 = time.perf_counter()
for i, x in enumerate(x_test):
    t_i = time.perf_counter()
    enc[i] = model(model.input(ctx, x.tolist())).decrypt()[0]
    print(f"  sample {i + 1}/{N_TEST}: {time.perf_counter() - t_i:.1f}s")
elapsed = time.perf_counter() - t0

print()
print("=== deep MLP (real ReLU train, f_n composite encrypted) ===")
print(f"{'plaintext R2':<26}: {r2_score(y_test, plain):.4f}")
print(f"{'encrypted R2':<26}: {r2_score(y_test, enc):.4f}")
print(f"{'enc vs plain max |err|':<26}: {np.abs(enc - plain).max():.6f}")
print(f"{'enc vs plain mean |err|':<26}: {np.abs(enc - plain).mean():.6f}")
print(f"{'latency':<26}: {elapsed / N_TEST:.1f} s/sample")
print(f"{'GPU memory (RMM pool)':<26}: {gpu_mem_mib()} MiB")
