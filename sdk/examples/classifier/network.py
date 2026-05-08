"""Shared network definition and dataset for all bench-dev scripts.

Architecture: Input(64) -> Linear(64->64) -> Square (x^2) -> Linear(64->10)

Square activation costs exactly 1 CKKS multiplication level,
making it the natural FHE-compatible non-linearity for this benchmark.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from torch.utils.data import TensorDataset, DataLoader

IN_FEATURES = 64
HIDDEN      = 64
N_CLASSES   = 10
FHE_BATCH   = 10
RANDOM_SEED = 42

np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)


class SquareNet(nn.Module):
    """Two-layer FC net with square activation — identical topology used by all benchmarks."""

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(IN_FEATURES, HIDDEN)
        self.fc2 = nn.Linear(HIDDEN, N_CLASSES)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.fc1(x) ** 2)


def load_dataset():
    """Return (X_train, X_test, y_train, y_test, X_fhe, y_fhe).

    X_fhe / y_fhe is the first FHE_BATCH samples of the test set,
    used as the encrypted inference batch.
    """
    X, y = make_classification(
        n_samples=1000, n_features=IN_FEATURES, n_classes=N_CLASSES,
        n_informative=30, n_redundant=10, n_repeated=0,
        random_state=RANDOM_SEED,
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y,
    )
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)
    X_fhe   = X_test[:FHE_BATCH]
    y_fhe   = y_test[:FHE_BATCH]
    return X_train, X_test, y_train, y_test, X_fhe, y_fhe


def train_squarenet(X_train, y_train, epochs=30, lr=1e-3, batch_size=64) -> SquareNet:
    model   = SquareNet()
    X_t     = torch.tensor(X_train, dtype=torch.float32)
    y_t     = torch.tensor(y_train, dtype=torch.long)
    loader  = DataLoader(TensorDataset(X_t, y_t), batch_size=batch_size, shuffle=True)
    opt     = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    model.train()
    for _ in range(epochs):
        for xb, yb in loader:
            opt.zero_grad()
            loss_fn(model(xb), yb).backward()
            opt.step()
    return model


def plaintext_accuracy(model: SquareNet, X, y) -> float:
    model.eval()
    with torch.no_grad():
        preds = model(torch.tensor(X, dtype=torch.float32)).argmax(1).numpy()
    return accuracy_score(y, preds)


def plaintext_logits(model: SquareNet, x: list) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        return model(torch.tensor(x, dtype=torch.float32)).numpy()


def extract_weights(model: SquareNet):
    """Return (W1, b1, W2, b2) as plain Python lists.

    W1: [HIDDEN, IN_FEATURES]  — shape [64, 64]
    W2: [N_CLASSES, HIDDEN]    — shape [10, 64]
    """
    W1 = model.fc1.weight.detach().numpy().tolist()
    b1 = model.fc1.bias.detach().numpy().tolist()
    W2 = model.fc2.weight.detach().numpy().tolist()
    b2 = model.fc2.bias.detach().numpy().tolist()
    return W1, b1, W2, b2
