"""Plaintext training for the CNN classifier (cross-entropy + Adam)."""

import os

import torch
from torch import nn

from .data import Dataset
from .networks import build_shallow_cnn

_CACHE_DIR = os.path.join(os.path.dirname(__file__), "trained")
_BUILDERS = {"shallow_cnn": build_shallow_cnn}


def train_model(
    model: nn.Module,
    data: Dataset,
    epochs: int = 8,
    batch_size: int = 256,
    lr: float = 1e-3,
) -> nn.Module:
    """Train `model` with mini-batch Adam + cross-entropy on MNIST."""
    h, w = data.image_shape
    x = torch.tensor(data.x_train, dtype=torch.float32).reshape(-1, 1, h, w)
    y = torch.tensor(data.y_train, dtype=torch.long)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    n = len(x)
    model.train()
    for _ in range(epochs):
        perm = torch.randperm(n)
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            opt.zero_grad()
            loss_fn(model(x[idx]), y[idx]).backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
    model.eval()
    return model


def get_trained_model(name: str, data: Dataset) -> nn.Module:
    """Return a trained network, caching weights on first call."""
    model = _BUILDERS[name]()
    path = os.path.join(_CACHE_DIR, f"{name}.pt")
    if os.path.exists(path):
        model.load_state_dict(torch.load(path))
        return model.eval()
    train_model(model, data)
    os.makedirs(_CACHE_DIR, exist_ok=True)
    torch.save(model.state_dict(), path)
    return model
