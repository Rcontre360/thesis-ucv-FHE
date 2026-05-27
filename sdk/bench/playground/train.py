import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model import SEED, N_TEST, N_CALIB, build_network, load_synth, train_model
from shared.io import save_weights, save_inputs

CASE_DIR = os.path.dirname(os.path.abspath(__file__))


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data = load_synth()
    model = build_network().to(device)
    train_model(model, data, device)

    x_test = data.x_test[:N_TEST]
    with torch.no_grad():
        float_logits = model(torch.tensor(x_test, dtype=torch.float32, device=device)).cpu().numpy()

    save_weights(model, CASE_DIR)
    save_inputs(
        CASE_DIR,
        x_test=x_test,
        y_test=data.y_test[:N_TEST],
        x_calib=data.x_train[:N_CALIB],
        float_logits=float_logits,
    )
    print("saved artifacts to", os.path.join(CASE_DIR, "artifacts"))


if __name__ == "__main__":
    main()
