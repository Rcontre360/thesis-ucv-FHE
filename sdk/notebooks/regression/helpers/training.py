"""Plaintext training: a standard regression loop plus a train-or-load cache.

The per-library benchmark files call `get_trained_model` — it trains the
network once (with real ReLU), caches the weights, and returns the frozen
model on later runs.
"""

import os

import torch
from torch import nn

from .data import Dataset
from .networks import build_shallow, build_deep

_CACHE_DIR = os.path.join(os.path.dirname(__file__), "trained")
_BUILDERS = {"shallow": build_shallow, "deep": build_deep}


def train_model(
    model: nn.Module,
    data: Dataset,
    epochs: int = 400,
    batch_size: int = 256,
    lr: float = 1e-3,
) -> nn.Module:
    """Train `model` with mini-batch Adam + MSE on the standardized data."""
    x = torch.tensor(data.x_train, dtype=torch.float32)
    y = torch.tensor(data.y_train, dtype=torch.float32).reshape(-1, 1)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
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
    """Return a trained network ("shallow" or "deep"), caching it on first call."""
    model = _BUILDERS[name](data.n_features)
    path = os.path.join(_CACHE_DIR, f"{name}.pt")
    if os.path.exists(path):
        model.load_state_dict(torch.load(path))
        return model.eval()
    train_model(model, data)
    os.makedirs(_CACHE_DIR, exist_ok=True)
    torch.save(model.state_dict(), path)
    return model
