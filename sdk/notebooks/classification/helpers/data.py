"""MNIST dataset: fetch, normalize, train/test split.

Real-world classification — 28x28 grayscale handwritten digits, 10 classes.
Standardized globally with train statistics so the CNN trains stably and the
HE side sees inputs in a known range.
"""

from dataclasses import dataclass
from typing import Tuple

import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

SEED = 42


@dataclass
class Dataset:
    """A standardized MNIST train/test split."""

    x_train: np.ndarray   # (N, 784) standardized, float32
    x_test: np.ndarray
    y_train: np.ndarray   # int labels in [0, 9]
    y_test: np.ndarray
    image_shape: Tuple[int, int] = (28, 28)
    n_classes: int = 10

    @property
    def n_features(self) -> int:
        return self.x_train.shape[1]


def load_mnist(test_size: float = 0.2) -> Dataset:
    """Fetch MNIST via `fetch_openml`, scale to [0, 1], standardize, and split.

    First call downloads ~75 MB; subsequent calls hit the sklearn cache.
    """
    raw = fetch_openml("mnist_784", version=1, as_frame=False, parser="auto")
    x = raw.data.astype(np.float32) / 255.0
    y = raw.target.astype(np.int64)
    x_tr, x_te, y_tr, y_te = train_test_split(
        x, y, test_size=test_size, random_state=SEED, stratify=y
    )
    # Standardize with train statistics so test sees the same scale.
    mean, std = float(x_tr.mean()), float(x_tr.std())
    x_tr = (x_tr - mean) / std
    x_te = (x_te - mean) / std
    return Dataset(x_tr, x_te, y_tr, y_te)
