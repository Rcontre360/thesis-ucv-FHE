import numpy as np
import torch

from bench.playground.model import build_network
from bench.shared.io import load_weights, load_inputs
from bench.shared.measure import Measure, Timer
from bench.shared.metrics import accuracy
from bench.shared.runner import emit


def run(case_dir: str) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = load_inputs(case_dir)
    x_test = data["x_test"].astype(np.float32)
    y_test = data["y_test"]
    n_test = len(x_test)

    model = load_weights(build_network(), case_dir, device=device).to(device).eval()

    alloc_probe = torch.cuda.memory_allocated if device.type == "cuda" else None
    with Measure(alloc_probe=alloc_probe) as mem:
        with torch.no_grad(), Timer() as t_infer:
            x = torch.tensor(x_test, dtype=torch.float32, device=device)
            logits = model(x).cpu().numpy()
    acc = accuracy(logits.argmax(axis=1), y_test)

    emit({
        "backend": "pytorch_plain",
        "float_accuracy": acc,
        "approx_accuracy": acc,
        "accuracy": acc,
        "agreement": 1.0,
        "output_mae": 0.0,
        "precision_bits": None,
        "keygen_s": 0.0,
        "compile_s": 0.0,
        "latency_s": t_infer.elapsed_s / n_test,
        "vram_mb": mem.peak_vram_mb,
        "vram_alloc_mb": mem.peak_alloc_mb,
        "ram_mb": mem.peak_rss_mb,
    })
