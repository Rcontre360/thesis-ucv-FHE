import os

import numpy as np
import torch

from bench.shared.config import SDK_ROOT, resolve_samples
from bench.mlp.model import N_FEATURES, build_network
from bench.shared.io import emit, load_weights, load_inputs
from bench.shared.measure import Measure, Timer, phase_metrics
from bench.shared.metrics import r2_score, pred_fidelity

import orion
import orion.nn as on


class OrionNet(on.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc1 = on.Linear(N_FEATURES, 128)
        self.act1 = on.ReLU()
        self.fc2 = on.Linear(128, 64)
        self.act2 = on.ReLU()
        self.fc3 = on.Linear(64, 1)

    def forward(self, x):
        return self.fc3(self.act2(self.fc2(self.act1(self.fc1(x)))))


def run(case_dir: str) -> None:
    counts = resolve_samples("orion")
    # ReLU in Orion is a deep Chebyshev expansion; the resnet config has the
    # bootstrap parameters configured for that depth.
    orion_config = os.path.join(SDK_ROOT, "temp", "orion", "configs", "resnet.yml")

    data = load_inputs(case_dir)
    x_test = data["x_test"].astype(np.float32)
    y_test = data["y_test"]
    x_calib = data["x_calib"]
    float_preds = data["float_preds"]

    x_lat = x_test[:counts.latency]
    y_lat = y_test[:counts.latency]
    x_acc = x_test[:counts.accuracy]
    y_acc = y_test[:counts.accuracy]
    float_lat = float_preds[:counts.latency]

    model = load_weights(build_network(), case_dir).cpu().eval()

    orion_net = OrionNet()
    with torch.no_grad():
        orion_net.fc1.weight.copy_(model[0].weight.detach())
        orion_net.fc1.bias.copy_(model[0].bias.detach())
        orion_net.fc2.weight.copy_(model[2].weight.detach())
        orion_net.fc2.bias.copy_(model[2].bias.detach())
        orion_net.fc3.weight.copy_(model[4].weight.detach())
        orion_net.fc3.bias.copy_(model[4].bias.detach())
    orion_net.eval()

    enc_preds = np.empty(counts.latency, dtype=np.float64)

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
            enc_preds[i] = float(np.asarray(out).reshape(-1)[0])

    orion_net.eval()
    with torch.no_grad(), Timer() as t_acc:
        acc_preds = orion_net(torch.tensor(x_acc, dtype=torch.float32)).cpu().numpy().reshape(-1)
    approx_r2 = r2_score(y_acc, acc_preds)

    output_mae, precision = pred_fidelity(float_lat, enc_preds)

    emit({
        "backend": "orion",
        "float_r2": r2_score(y_test, float_preds),
        "approx_r2": approx_r2,
        "r2": r2_score(y_lat, enc_preds),
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
