import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model import N_CLASSES, build_network
from shared.io import load_weights, load_inputs
from shared.measure import Measure, Timer
from shared.metrics import accuracy
from shared.runner import emit

CASE_DIR = os.path.dirname(os.path.abspath(__file__))


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = load_inputs(CASE_DIR)
    x_test = data["x_test"].astype(np.float32)
    y_test = data["y_test"]
    n_test = len(x_test)

    model = load_weights(build_network(), CASE_DIR, device=device).to(device).eval()

    alloc_probe = torch.cuda.memory_allocated if device.type == "cuda" else None
    with Measure(alloc_probe=alloc_probe) as mem:
        with torch.no_grad(), Timer() as t_infer:
            x = torch.tensor(x_test, dtype=torch.float32, device=device)
            logits = model(x).cpu().numpy()
    pred = logits.argmax(axis=1)

    emit({
        "backend": "pytorch_plain",
        "accuracy": accuracy(pred, y_test),
        "keygen_s": 0.0,
        "compile_s": 0.0,
        "latency_s": t_infer.elapsed_s / n_test,
        "vram_mb": mem.peak_vram_mb,
        "vram_alloc_mb": mem.peak_alloc_mb,
        "ram_mb": mem.peak_rss_mb,
    })


if __name__ == "__main__":
    main()
