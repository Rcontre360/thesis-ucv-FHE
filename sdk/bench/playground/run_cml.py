import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model import SEED, N_CLASSES, build_network
from shared.io import load_weights, load_inputs
from shared.measure import Measure, Timer
from shared.metrics import accuracy
from shared.runner import emit

from concrete.ml.torch.compile import compile_torch_model
import concrete.compiler

CASE_DIR = os.path.dirname(os.path.abspath(__file__))

N_BITS = 6


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available")
    if not concrete.compiler.check_gpu_enabled():
        raise RuntimeError("concrete-python built without GPU support")
    if not concrete.compiler.check_gpu_available():
        raise RuntimeError("Concrete-ML cannot detect the GPU")

    data = load_inputs(CASE_DIR)
    x_test = data["x_test"].astype(np.float32)
    y_test = data["y_test"]
    calib = torch.tensor(data["x_calib"], dtype=torch.float32)
    n_test = len(x_test)

    model = load_weights(build_network(), CASE_DIR).cpu().eval()
    pred = np.empty(n_test, dtype=int)

    with Measure() as mem:
        with Timer() as t_compile:
            cml_model = compile_torch_model(
                model, torch_inputset=calib, n_bits=N_BITS, device="cuda",
            )

        with Timer() as t_keygen:
            cml_model.fhe_circuit.keygen(force=True, seed=SEED)

        with Timer() as t_infer:
            for i, x in enumerate(x_test):
                logits = cml_model.forward(x.reshape(1, -1), fhe="execute").reshape(-1)[:N_CLASSES]
                pred[i] = int(np.argmax(logits))

    emit({
        "backend": "concrete-ml",
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
