import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model import N_FEATURES, N_CLASSES, build_network
from shared.io import load_weights, load_inputs
from shared.measure import Measure, Timer
from shared.metrics import accuracy
from shared.runner import emit

import orion
import orion.nn as on

CASE_DIR = os.path.dirname(os.path.abspath(__file__))
ORION_CONFIG = os.path.abspath(
    os.path.join(CASE_DIR, "..", "..", "temp", "orion", "configs", "mlp.yml")
)


class OrionNet(on.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = on.Linear(N_FEATURES, 4)
        self.act = on.Quad()
        self.fc2 = on.Linear(4, N_CLASSES)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


def main():
    data = load_inputs(CASE_DIR)
    x_test = data["x_test"].astype(np.float32)
    y_test = data["y_test"]
    x_calib = data["x_calib"]
    n_test = len(x_test)

    model = load_weights(build_network(), CASE_DIR).cpu().eval()

    orion_net = OrionNet()
    with torch.no_grad():
        orion_net.fc1.weight.copy_(model[0].weight.detach())
        orion_net.fc1.bias.copy_(model[0].bias.detach())
        orion_net.fc2.weight.copy_(model[2].weight.detach())
        orion_net.fc2.bias.copy_(model[2].bias.detach())
    orion_net.eval()

    pred = np.empty(n_test, dtype=int)
    with Measure() as mem:
        with Timer() as t_keygen:
            orion.init_scheme(ORION_CONFIG)

        with Timer() as t_compile:
            orion.fit(orion_net, torch.tensor(x_calib[:256], dtype=torch.float32), batch_size=64)
            input_level = orion.compile(orion_net)

        with Timer() as t_infer:
            for i, x in enumerate(x_test):
                ctxt = orion.encrypt(
                    orion.encode(torch.tensor(x.reshape(1, -1), dtype=torch.float32), input_level)
                )
                orion_net.he()
                out = orion_net(ctxt).decrypt().decode()
                pred[i] = int(np.asarray(out).reshape(-1)[:N_CLASSES].argmax())

    emit({
        "backend": "orion",
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
