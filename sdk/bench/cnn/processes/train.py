import os

import numpy as np
import torch

from bench.shared.config import SEED, N_CALIB
from bench.shared.training import train_model
from bench.cnn.model import build_network, load_mnist, TRAIN_CONFIG, CHANNELS, IMAGE_SHAPE
from bench.shared.io import save_weights, save_inputs


def run(case_dir: str) -> None:
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data = load_mnist()
    model = build_network().to(device)
    train_model(model, data, device, TRAIN_CONFIG)

    with torch.no_grad():
        x = torch.tensor(data.x_test, dtype=torch.float32, device=device)
        x = x.reshape(-1, CHANNELS, *IMAGE_SHAPE)
        float_logits = model(x).cpu().numpy()

    save_weights(model, case_dir)
    save_inputs(
        case_dir,
        x_test=data.x_test,
        y_test=data.y_test,
        x_calib=data.x_train[:N_CALIB],
        float_logits=float_logits,
    )
    print("saved artifacts to", os.path.join(case_dir, "artifacts"))
