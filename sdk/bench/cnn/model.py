import numpy as np
from torch import nn
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

from bench.shared.config import SEED
from bench.shared.training import Dataset, TrainConfig

IMAGE_SHAPE: tuple[int, int] = (28, 28)
CHANNELS: int = 1
N_CLASSES: int = 10
TEST_SIZE: float = 0.2

TRAIN_CONFIG: TrainConfig = TrainConfig(
    epochs=8, lr=1e-3, batch=256, grad_clip=5.0,
    input_shape=(CHANNELS, *IMAGE_SHAPE),
)


def build_network() -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_channels=CHANNELS, out_channels=4, kernel_size=7, stride=5),
        nn.ReLU(),
        nn.Flatten(),
        nn.Linear(4 * 5 * 5, N_CLASSES),
    )


def load_mnist() -> Dataset:
    # fetch_openml avoids the `datasets` package the notebook uses, keeping
    # the bench install footprint to what's already in tox.ini.
    raw = fetch_openml("mnist_784", version=1, as_frame=False, parser="auto")
    x = raw.data.astype(np.float32) / 255.0
    y = raw.target.astype(np.int64)
    x_tr, x_te, y_tr, y_te = train_test_split(
        x, y, test_size=TEST_SIZE, random_state=SEED, stratify=y,
    )
    mean, std = float(x_tr.mean()), float(x_tr.std())
    x_tr = ((x_tr - mean) / std).astype(np.float32)
    x_te = ((x_te - mean) / std).astype(np.float32)
    return Dataset(x_tr, x_te, y_tr, y_te)
