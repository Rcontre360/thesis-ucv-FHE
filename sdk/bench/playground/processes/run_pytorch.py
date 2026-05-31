import numpy as np
import torch

from bench.shared.config import resolve_samples
from bench.playground.model import build_network
from bench.shared.io import emit, load_weights, load_inputs
from bench.shared.measure import Measure, Timer, phase_metrics
from bench.shared.metrics import accuracy


def run(case_dir: str) -> None:
    counts = resolve_samples("pytorch")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = load_inputs(case_dir)
    x_test = data["x_test"].astype(np.float32)
    y_test = data["y_test"]

    x_lat = x_test[:counts.latency]
    x_acc = x_test[:counts.accuracy]
    y_acc = y_test[:counts.accuracy]

    model = load_weights(build_network(), case_dir, device=device).to(device).eval()
    alloc_probe = torch.cuda.memory_allocated if device.type == "cuda" else None

    with Measure(alloc_probe=alloc_probe) as m_infer:
        with torch.no_grad():
            lat_logits = model(torch.tensor(x_lat, dtype=torch.float32, device=device)).cpu().numpy()

    with torch.no_grad(), Timer() as t_acc:
        acc_logits = model(torch.tensor(x_acc, dtype=torch.float32, device=device)).cpu().numpy()
    acc = accuracy(acc_logits.argmax(axis=1), y_acc)

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
        "latency_s": m_infer.elapsed_s / counts.latency,
        "accuracy_per_sample_s": t_acc.elapsed_s / counts.accuracy,
        "latency_n": counts.latency,
        "accuracy_n": counts.accuracy,

        **phase_metrics({"keygen": None, "compile": None, "infer": m_infer}),
    })
