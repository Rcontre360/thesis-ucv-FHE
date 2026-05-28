import os

import numpy as np
import torch

from bench.shared.config import SEED, N_TEST, N_CALIB
from bench.playground.model import build_network, load_synth, train_model
from bench.shared.io import save_weights, save_inputs


def run(case_dir: str) -> None:
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data = load_synth()
    model = build_network().to(device)
    train_model(model, data, device)

    x_test = data.x_test[:N_TEST]
    with torch.no_grad():
        float_logits = model(torch.tensor(x_test, dtype=torch.float32, device=device)).cpu().numpy()

    save_weights(model, case_dir)
    save_inputs(
        case_dir,
        x_test=x_test,
        y_test=data.y_test[:N_TEST],
        x_calib=data.x_train[:N_CALIB],
        float_logits=float_logits,
    )
    print("saved artifacts to", os.path.join(case_dir, "artifacts"))
