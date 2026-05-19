"""California Housing dataset: download, split, standardize.

Real-world regression — predict a district's median house value from 8
features. Both benchmark networks share this data so their numbers are
comparable.
"""

from dataclasses import dataclass

import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

SEED = 42


@dataclass
class Dataset:
    """A standardized train/test split.

    Features are z-scored. The target is z-scored too (square-activation
    networks train more stably that way); `denormalize_target` converts a
    prediction back to the original $100k units.
    """

    x_train: np.ndarray
    x_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray
    y_mean: float
    y_std: float

    @property
    def n_features(self) -> int:
        return self.x_train.shape[1]

    def denormalize_target(self, y) -> np.ndarray:
        return np.asarray(y, dtype=float) * self.y_std + self.y_mean


def load_california_housing(test_size: float = 0.2) -> Dataset:
    """Fetch California Housing and return a standardized split.

    The scaler is fit on the train split only, then applied to both.
    """
    raw = fetch_california_housing()
    x_tr, x_te, y_tr, y_te = train_test_split(
        raw.data, raw.target, test_size=test_size, random_state=SEED
    )
    scaler = StandardScaler().fit(x_tr)
    x_tr, x_te = scaler.transform(x_tr), scaler.transform(x_te)

    y_mean, y_std = float(y_tr.mean()), float(y_tr.std())
    y_tr = (y_tr - y_mean) / y_std
    y_te = (y_te - y_mean) / y_std

    return Dataset(x_tr, x_te, y_tr, y_te, y_mean, y_std)
