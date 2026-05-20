"""Square activation — x -> x^2.

The simplest polynomial activation that fits any leveled HE scheme:
multiplicative depth 1, no parameters, no calibration. Used by CryptoNets
and serves as the cross-backend (CKKS / TFHE / our SDK) benchmark
activation.
"""

from typing import TYPE_CHECKING

import numpy as np

from core.layer import Layer

if TYPE_CHECKING:
    from api.ciphertext import EncryptedVector


class Square(Layer):
    """y = x * x. Consumes one multiplicative level; no parameters."""

    def __call__(self, x: "EncryptedVector") -> "EncryptedVector":
        x = x._context._prepare_for(x, 1)
        return x * x

    def mult_depth(self) -> int:
        return 1

    def forward_plain(self, x: np.ndarray) -> np.ndarray:
        return x ** 2
