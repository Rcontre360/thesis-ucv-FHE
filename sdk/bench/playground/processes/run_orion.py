import os

import numpy as np
import torch

from bench.shared.config import SDK_ROOT, resolve_samples
from bench.playground.model import N_FEATURES, N_CLASSES, build_network
from bench.shared.io import emit, load_weights, load_inputs
from bench.shared.measure import Measure, Timer, phase_metrics
from bench.shared.metrics import accuracy, fidelity

import orion
import orion.nn as on


class OrionNet(on.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc1 = on.Linear(N_FEATURES, 4)
        self.act = on.Quad()
        self.fc2 = on.Linear(4, N_CLASSES)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


def run(case_dir: str) -> None:
    counts = resolve_samples("orion")
    orion_config = os.path.join(SDK_ROOT, "temp", "orion", "configs", "mlp.yml")

    data = load_inputs(case_dir)
    x_test = data["x_test"].astype(np.float32)
    y_test = data["y_test"]
    x_calib = data["x_calib"]
    float_logits = data["float_logits"]

    x_lat = x_test[:counts.latency]
    x_acc = x_test[:counts.accuracy]
    y_acc = y_test[:counts.accuracy]
    float_lat = float_logits[:counts.latency]

    model = load_weights(build_network(), case_dir).cpu().eval()

    orion_net = OrionNet()
    with torch.no_grad():
        orion_net.fc1.weight.copy_(model[0].weight.detach())
        orion_net.fc1.bias.copy_(model[0].bias.detach())
        orion_net.fc2.weight.copy_(model[2].weight.detach())
        orion_net.fc2.bias.copy_(model[2].bias.detach())
    orion_net.eval()

    enc_logits = np.empty((counts.latency, N_CLASSES), dtype=np.float64)

    with Measure() as m_keygen:
        orion.init_scheme(orion_config)

    with Measure() as m_compile:
        orion.fit(orion_net, torch.tensor(x_calib[:256], dtype=torch.float32), batch_size=64)
        input_level = orion.compile(orion_net)

    with Measure() as m_infer:
        for i, x in enumerate(x_lat):
            ctxt = orion.encrypt(
                orion.encode(torch.tensor(x.reshape(1, -1), dtype=torch.float32), input_level)
            )
            orion_net.he()
            out = orion_net(ctxt).decrypt().decode()
            enc_logits[i] = np.asarray(out).reshape(-1)[:N_CLASSES]

    orion_net.eval()
    with torch.no_grad(), Timer() as t_acc:
        acc_logits = orion_net(torch.tensor(x_acc, dtype=torch.float32)).cpu().numpy()
    approx_accuracy = accuracy(acc_logits[:, :N_CLASSES].argmax(axis=1), y_acc)

    agreement, output_mae, precision = fidelity(float_lat, enc_logits)

    emit({
        "backend": "orion",
        "float_accuracy": accuracy(float_logits.argmax(axis=1), y_test),
        "approx_accuracy": approx_accuracy,
        "accuracy": accuracy(enc_logits.argmax(axis=1), y_test[:counts.latency]),
        "agreement": agreement,
        "output_mae": output_mae,
        "precision_bits": precision,

        "keygen_s": m_keygen.elapsed_s,
        "compile_s": m_compile.elapsed_s,
        "latency_s": m_infer.elapsed_s / counts.latency,
        "accuracy_per_sample_s": t_acc.elapsed_s / counts.accuracy,
        "latency_n": counts.latency,
        "accuracy_n": counts.accuracy,

        **phase_metrics({"keygen": m_keygen, "compile": m_compile, "infer": m_infer}),
    })
