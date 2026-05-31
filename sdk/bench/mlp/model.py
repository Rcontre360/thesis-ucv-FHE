import numpy as np
from torch import nn
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from bench.shared.config import SEED
from bench.shared.training import Dataset, TrainConfig

N_FEATURES: int = 8
TEST_SIZE: float = 0.2

TRAIN_CONFIG: TrainConfig = TrainConfig(epochs=400, lr=1e-3, batch=256)


def build_network() -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(N_FEATURES, 128),
        nn.ReLU(),
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Linear(64, 1),
    )


def load_california() -> Dataset:
    # R² is scale-invariant, so we keep y z-scored and don't carry y_mean/y_std around.
    raw = fetch_california_housing()
    x_tr, x_te, y_tr, y_te = train_test_split(
        raw.data, raw.target, test_size=TEST_SIZE, random_state=SEED,
    )
    scaler = StandardScaler().fit(x_tr)
    x_tr = scaler.transform(x_tr).astype(np.float32)
    x_te = scaler.transform(x_te).astype(np.float32)
    y_mean, y_std = float(y_tr.mean()), float(y_tr.std())
    y_tr = ((y_tr - y_mean) / y_std).astype(np.float32)
    y_te = ((y_te - y_mean) / y_std).astype(np.float32)
    return Dataset(x_tr, x_te, y_tr, y_te)
