import os

import numpy as np
import torch

from bench.shared.config import SDK_ROOT, resolve_samples
from bench.mlp.model import N_FEATURES, build_network
from bench.shared.io import emit, load_weights, load_inputs
from bench.shared.measure import Measure, phase_metrics
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

    # Single encrypted loop sized to counts.accuracy; counts.latency is 0 by
    # config because orion_net.eval() is NOT a clear-equivalent of the encrypted
    # path (see config.toml header).
    n = counts.accuracy
    x_acc = x_test[:n]
    y_acc = y_test[:n]
    float_acc = float_preds[:n]

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

    enc_preds = np.empty(n, dtype=np.float64)

    with Measure() as m_keygen:
        orion.init_scheme(orion_config)

    with Measure() as m_compile:
        # Orion bakes the calib batch dim into its matrix packing, so 200 samples
        # would blow fc1 up to a 32k×32k diagonal set. 8 samples is enough for
        # range estimation on the Chebyshev fit.
        orion.fit(orion_net, torch.tensor(x_calib[:8], dtype=torch.float32), batch_size=8)
        input_level = orion.compile(orion_net)

    with Measure() as m_infer:
        for i, x in enumerate(x_acc):
            ctxt = orion.encrypt(
                orion.encode(torch.tensor(x.reshape(1, -1), dtype=torch.float32), input_level)
            )
            orion_net.he()
            out = orion_net(ctxt).decrypt().decode()
            enc_preds[i] = float(np.asarray(out).reshape(-1)[0])

    per_sample_s = m_infer.elapsed_s / n
    encrypted_r2 = r2_score(y_acc, enc_preds)
    output_mae, precision = pred_fidelity(float_acc, enc_preds)

    emit({
        "backend": "orion",
        "float_r2": r2_score(y_test, float_preds),
        # approx_r2 and r2 are equal here: the same encrypted loop produces
        # both. Both columns retained for cross-backend table consistency.
        "approx_r2": encrypted_r2,
        "r2": encrypted_r2,
        "output_mae": output_mae,
        "precision_bits": precision,

        "keygen_s": m_keygen.elapsed_s,
        "compile_s": m_compile.elapsed_s,
        "latency_s": per_sample_s,
        "accuracy_per_sample_s": per_sample_s,
        "latency_n": counts.latency,  # 0 — no separate latency loop, see config
        "accuracy_n": counts.accuracy,

        **phase_metrics({"keygen": m_keygen, "compile": m_compile, "infer": m_infer}),
    })
