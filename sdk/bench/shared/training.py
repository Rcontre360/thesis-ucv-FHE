from dataclasses import dataclass

import numpy as np
import torch
from torch import nn


@dataclass
class Dataset:
    x_train: np.ndarray
    x_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray


@dataclass
class TrainConfig:
    epochs: int = 30
    lr: float = 1e-2
    batch: int = 64
    grad_clip: float = 0.0
    input_shape: tuple[int, ...] | None = None


def train_model(model: nn.Module, data: Dataset, device: torch.device,
                config: TrainConfig = TrainConfig()) -> nn.Module:
    x = torch.tensor(data.x_train, dtype=torch.float32, device=device)
    if config.input_shape is not None:
        x = x.reshape(-1, *config.input_shape)
    y = torch.tensor(data.y_train, dtype=torch.long, device=device)
    opt = torch.optim.Adam(model.parameters(), lr=config.lr)
    loss_fn = nn.CrossEntropyLoss()
    n = len(x)
    model.train()
    for _ in range(config.epochs):
        perm = torch.randperm(n, device=device)
        for i in range(0, n, config.batch):
            idx = perm[i:i + config.batch]
            opt.zero_grad()
            loss = loss_fn(model(x[idx]), y[idx])
            loss.backward()
            if config.grad_clip > 0.0:
                nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            opt.step()
    model.eval()
    return model


def train_regression(model: nn.Module, data: Dataset, device: torch.device,
                     config: TrainConfig = TrainConfig()) -> nn.Module:
    x = torch.tensor(data.x_train, dtype=torch.float32, device=device)
    y = torch.tensor(data.y_train, dtype=torch.float32, device=device).reshape(-1, 1)
    opt = torch.optim.Adam(model.parameters(), lr=config.lr)
    loss_fn = nn.MSELoss()
    n = len(x)
    model.train()
    for _ in range(config.epochs):
        perm = torch.randperm(n, device=device)
        for i in range(0, n, config.batch):
            idx = perm[i:i + config.batch]
            opt.zero_grad()
            loss = loss_fn(model(x[idx]), y[idx])
            loss.backward()
            opt.step()
    model.eval()
    return model
