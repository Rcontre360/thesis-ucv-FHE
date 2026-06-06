import os

import numpy as np
import torch

from bench.shared.config import SDK_ROOT, resolve_samples
from bench.cnn.model import build_network, N_CLASSES, CHANNELS, IMAGE_SHAPE
from bench.shared.io import emit, load_weights, load_inputs
from bench.shared.measure import Measure, phase_metrics
from bench.shared.metrics import accuracy, fidelity

import orion
import orion.nn as on


class OrionNet(on.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = on.Conv2d(CHANNELS, 4, kernel_size=7, stride=5)
        self.act = on.ReLU()
        self.flat = on.Flatten()
        self.fc = on.Linear(4 * 5 * 5, N_CLASSES)

    def forward(self, x):
        return self.fc(self.flat(self.act(self.conv(x))))


def run(case_dir: str) -> None:
    counts = resolve_samples("orion")
    # resnet.yml has the bootstrap chain provisioned for deep ReLU networks.
    orion_config = os.path.join(SDK_ROOT, "temp", "orion", "configs", "resnet.yml")

    h, w = IMAGE_SHAPE
    data = load_inputs(case_dir)
    x_test = data["x_test"].astype(np.float32)
    y_test = data["y_test"]
    x_calib = data["x_calib"]
    float_logits = data["float_logits"]

    # Single encrypted loop sized to counts.accuracy; counts.latency is 0 by
    # config because orion_net.eval() is NOT a clear-equivalent of the encrypted
    # path (see config.toml header).
    n = counts.accuracy
    x_acc = x_test[:n]
    y_acc = y_test[:n]
    float_acc = float_logits[:n]

    model = load_weights(build_network(), case_dir).cpu().eval()

    orion_net = OrionNet()
    with torch.no_grad():
        orion_net.conv.weight.copy_(model[0].weight.detach())
        orion_net.conv.bias.copy_(model[0].bias.detach())
        orion_net.fc.weight.copy_(model[3].weight.detach())
        orion_net.fc.bias.copy_(model[3].bias.detach())
    orion_net.eval()

    enc_logits = np.empty((n, N_CLASSES), dtype=np.float64)

    with Measure() as m_keygen:
        orion.init_scheme(orion_config)

    with Measure() as m_compile:
        # See bench/mlp/processes/run_orion.py for why the calib batch is tiny:
        # Orion bakes batch into its matrix packing.
        calib = torch.tensor(x_calib[:8], dtype=torch.float32).reshape(-1, CHANNELS, h, w)
        orion.fit(orion_net, calib, batch_size=8)
        input_level = orion.compile(orion_net)

    with Measure() as m_infer:
        for i, x in enumerate(x_acc):
            img = torch.tensor(x.reshape(1, CHANNELS, h, w), dtype=torch.float32)
            ctxt = orion.encrypt(orion.encode(img, input_level))
            orion_net.he()
            out = orion_net(ctxt).decrypt().decode()
            enc_logits[i] = np.asarray(out).reshape(-1)[:N_CLASSES]

    per_sample_s = m_infer.elapsed_s / n
    enc_top1 = accuracy(enc_logits.argmax(axis=1), y_acc)
    agreement, output_mae, precision = fidelity(float_acc, enc_logits)

    emit({
        "backend": "orion",
        "float_accuracy": accuracy(float_logits.argmax(axis=1), y_test),
        # approx_accuracy and accuracy are equal here: the same encrypted loop
        # produces both. Both columns retained for cross-backend table consistency.
        "approx_accuracy": enc_top1,
        "accuracy": enc_top1,
        "agreement": agreement,
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
