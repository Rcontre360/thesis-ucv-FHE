import numpy as np
import torch
from torch import nn
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from bench.shared.config import SEED
from bench.shared.training import Dataset

N_FEATURES = 8
N_CLASSES = 2
N_SAMPLES = 1000


class Square(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * x


def build_network() -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(N_FEATURES, 4),
        Square(),
        nn.Linear(4, N_CLASSES),
    )


def load_synth() -> Dataset:
    x, y = make_classification(
        n_samples=N_SAMPLES, n_features=N_FEATURES, n_classes=N_CLASSES,
        n_informative=6, n_redundant=0, n_repeated=0,
        random_state=SEED,
    )
    x_tr, x_te, y_tr, y_te = train_test_split(
        x, y, test_size=0.2, random_state=SEED, stratify=y,
    )
    sc = StandardScaler().fit(x_tr)
    x_tr = sc.transform(x_tr).astype(np.float32)
    x_te = sc.transform(x_te).astype(np.float32)
    return Dataset(x_tr, x_te, y_tr.astype(np.int64), y_te.astype(np.int64))
