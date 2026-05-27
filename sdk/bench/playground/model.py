from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

SEED = 30391915
N_FEATURES = 8
N_CLASSES = 2
N_SAMPLES = 1000
N_TEST = 5
N_CALIB = 200


class Square(nn.Module):
    def forward(self, x):
        return x * x


def build_network():
    return nn.Sequential(
        nn.Linear(N_FEATURES, 4),
        Square(),
        nn.Linear(4, N_CLASSES),
    )


@dataclass
class Dataset:
    x_train: np.ndarray
    x_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray
    n_classes: int = N_CLASSES

    @property
    def n_features(self):
        return self.x_train.shape[1]


def load_synth():
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

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


def train_model(model, data, device, epochs=30, lr=1e-2, batch=64):
    x = torch.tensor(data.x_train, dtype=torch.float32, device=device)
    y = torch.tensor(data.y_train, dtype=torch.long, device=device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    n = len(x)
    model.train()
    for _ in range(epochs):
        perm = torch.randperm(n, device=device)
        for i in range(0, n, batch):
            idx = perm[i:i + batch]
            opt.zero_grad()
            loss = loss_fn(model(x[idx]), y[idx])
            loss.backward()
            opt.step()
    model.eval()
    return model
