"""Classification benchmark — shallow CNN on MNIST, our CKKS SDK.

Trains a depth-101 CNN on MNIST in plaintext (PyTorch), ports the frozen
weights into the SDK, calibrates per-neuron activation ranges on the training
data, runs encrypted inference, and reports accuracy. The 10 logits are
decrypted client-side and argmaxed there — softmax never enters the FHE
circuit.

Same depth as `notebooks/regression/sdk_shallow.py`; only the first layer
changes from Linear to Conv2D and the loss/metric changes from MSE/R^2 to
cross-entropy/accuracy. INSECURE parameters — demo only, like the regression
benchmarks.
"""

import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch

from helpers.data import load_mnist
from helpers.training import get_trained_model

from api import FHEContext
from api.layers.linear import Linear
from api.layers.conv2d import Conv2D
from api.functions.relu import ReLU
from api.sequential import Sequential
from core.enums import SecurityLevel

RELU_DEGREES = (5,) * 12
N_TEST = 5
COEFF_MODULUS = [60] + [50] * 28 + [60]


def to_sdk_model(torch_model: torch.nn.Module, image_shape) -> Sequential:
    """Port the trained PyTorch CNN into an SDK Sequential.

    Conv2d weights are (out, in, kH, kW) in both PyTorch and the SDK — direct
    copy. `Flatten` is a no-op for our SDK (Conv2D already outputs flat).
    """
    h, w = image_shape
    layers = []
    for m in torch_model:
        if isinstance(m, torch.nn.Conv2d):
            weight = m.weight.detach().numpy()
            bias = m.bias.detach().numpy()
            stride = m.stride[0] if isinstance(m.stride, tuple) else m.stride
            layers.append(Conv2D(
                in_channels=m.in_channels,
                out_channels=m.out_channels,
                kernel_size=m.kernel_size,
                input_shape=(h, w),
                weight=weight,
                bias=bias,
                stride=stride,
            ))
            kh, kw = m.kernel_size
            h = (h - kh) // stride + 1
            w = (w - kw) // stride + 1
        elif isinstance(m, torch.nn.Linear):
            weight = m.weight.detach().numpy()
            bias = m.bias.detach().numpy()
            layers.append(Linear(weight.shape[1], weight.shape[0], weight, bias))
        elif isinstance(m, torch.nn.Flatten):
            continue  # SDK Conv2D output is already flat
        else:  # nn.ReLU
            layers.append(ReLU(degrees=RELU_DEGREES))
    return Sequential(layers)


def gpu_mem_mib() -> int:
    out = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"]
    )
    return int(out.decode().split("\n")[0])


print("loading MNIST + training plaintext CNN...")
data = load_mnist()
torch_model = get_trained_model("shallow_cnn", data)

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

model = to_sdk_model(torch_model, data.image_shape)
print(f"network: {len(model._layers)} layers, mult_depth {sum(l.mult_depth() for l in model._layers)}")

print("compiling (calibrating + setting up SLIM bootstrapping)...")
t0 = time.perf_counter()
model.compile(ctx, calibration_data=data.x_train)
print(f"  compile: {time.perf_counter() - t0:.1f}s")
print(f"  bootstrapping enabled: {ctx._bootstrapping_ready}")

x_test = data.x_test[:N_TEST]
y_test = data.y_test[:N_TEST]
h, w = data.image_shape

# Plaintext reference (argmax of CNN logits).
with torch.no_grad():
    plain_logits = torch_model(
        torch.tensor(x_test, dtype=torch.float32).reshape(-1, 1, h, w)
    ).numpy()
plain_pred = plain_logits.argmax(axis=1)

print(f"running encrypted inference on {N_TEST} samples...")
enc_pred = np.empty(N_TEST, dtype=int)
t0 = time.perf_counter()
for i, x in enumerate(x_test):
    t_i = time.perf_counter()
    image = x.reshape(h, w)
    logits = model(model.input(ctx, image.tolist())).decrypt()[:data.n_classes]
    enc_pred[i] = int(np.argmax(logits))
    print(f"  sample {i+1}/{N_TEST}: {time.perf_counter()-t_i:.1f}s  "
          f"(label={y_test[i]}, plain={plain_pred[i]}, enc={enc_pred[i]})")
elapsed = time.perf_counter() - t0

print()
print("=== shallow CNN on MNIST — our SDK (CKKS, GPU) ===")
print(f"{'plaintext accuracy':<26}: {(plain_pred == y_test).mean():.2%}")
print(f"{'encrypted accuracy':<26}: {(enc_pred == y_test).mean():.2%}")
print(f"{'enc vs plain agreement':<26}: {(enc_pred == plain_pred).mean():.2%}")
print(f"{'latency':<26}: {elapsed / N_TEST:.1f} s/sample")
print(f"{'GPU memory (RMM pool)':<26}: {gpu_mem_mib()} MiB")
