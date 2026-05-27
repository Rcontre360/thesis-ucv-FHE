import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model import N_CLASSES, Square, build_network
from shared.io import load_weights, load_inputs
from shared.measure import Measure, Timer
from shared.metrics import accuracy
from shared.runner import emit

from api import FHEContext
from api.layers.linear import Linear
from api.functions.square import Square as SDKSquare
from api.sequential import Sequential
from core.enums import SecurityLevel
from core._backend import device_pool_used_bytes

CASE_DIR = os.path.dirname(os.path.abspath(__file__))

POLY_DEGREE = 8192
COEFF_MODULUS = [60, 40, 40, 40, 40, 60, 60]
SCALE = 2 ** 40
SECURITY = SecurityLevel.NONE


def to_sdk_model(model):
    layers = []
    for m in model.cpu():
        if isinstance(m, torch.nn.Linear):
            layers.append(Linear(
                m.in_features, m.out_features,
                m.weight.detach().numpy(), m.bias.detach().numpy(),
            ))
        elif isinstance(m, Square):
            layers.append(SDKSquare())
        else:
            raise TypeError(type(m).__name__)
    return Sequential(layers)


def main():
    data = load_inputs(CASE_DIR)
    x_test = data["x_test"].astype(np.float32)
    y_test = data["y_test"]
    n_test = len(x_test)

    model = load_weights(build_network(), CASE_DIR).eval()
    pred = np.empty(n_test, dtype=int)

    with Measure(alloc_probe=device_pool_used_bytes) as mem:
        with Timer() as t_keygen:
            ctx = (FHEContext()
                .set_poly_modulus_degree(POLY_DEGREE)
                .set_coeff_modulus_bit_sizes(COEFF_MODULUS)
                .set_scale(SCALE)
                .set_security_level(SECURITY)
                .build())

        with Timer() as t_compile:
            sdk_model = to_sdk_model(model).compile(ctx)

        with Timer() as t_infer:
            for i, x in enumerate(x_test):
                logits = sdk_model(sdk_model.input(ctx, x.tolist())).decrypt()[:N_CLASSES]
                pred[i] = int(np.argmax(logits))

    emit({
        "backend": "sdk",
        "accuracy": accuracy(pred, y_test),
        "keygen_s": t_keygen.elapsed_s,
        "compile_s": t_compile.elapsed_s,
        "latency_s": t_infer.elapsed_s / n_test,
        "vram_mb": mem.peak_vram_mb,
        "vram_alloc_mb": mem.peak_alloc_mb,
        "ram_mb": mem.peak_rss_mb,
    })


if __name__ == "__main__":
    main()
