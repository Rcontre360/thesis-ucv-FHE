from typing import Tuple

import numpy as np
import torch.nn as nn

from fhe_ml.ckks.containers.ciphertext import EncryptedVector
from fhe_ml.layers.base import Layer


class Square(Layer):
    """y = x * x. Consumes one multiplicative level; no parameters."""

    def __call__(self, x: EncryptedVector) -> EncryptedVector:
        x = x._context._prepare_for(x, 1)
        return x * x

    def mult_depth(self) -> int:
        return 1

    def forward_plain(self, x: np.ndarray) -> np.ndarray:
        return x ** 2

    @classmethod
    def from_torch(
        cls,
        module: nn.Module,
        input_shape: Tuple[int, ...],
    ) -> Tuple["Square", Tuple[int, ...]]:
        return cls(), input_shape
