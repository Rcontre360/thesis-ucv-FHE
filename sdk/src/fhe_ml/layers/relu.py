from math import factorial
from typing import TYPE_CHECKING, List, Optional, Tuple

import numpy as np
import torch.nn as nn

from fhe_ml.layers.base import Layer

if TYPE_CHECKING:
    from fhe_ml.ckks.containers.ciphertext import EncryptedVector


def _fn_coeffs(degree: int) -> np.ndarray:
    if degree < 3 or degree % 2 == 0:
        raise ValueError(f"f_n degree must be odd and >= 3; got {degree}")
    n = (degree - 1) // 2
    A = np.zeros((n + 1, n + 1))
    b = np.zeros(n + 1)
    b[0] = 1.0
    for k in range(n + 1):
        for j in range(n + 1):
            if j == 0:
                A[j, k] = 1.0
            elif j <= 2 * k + 1:
                A[j, k] = factorial(2 * k + 1) // factorial(2 * k + 1 - j)
    odd_coeffs = np.linalg.solve(A, b)
    coeffs = np.zeros(degree + 1)
    coeffs[1::2] = odd_coeffs
    return coeffs


class ReLU(Layer):
    """Polynomial ReLU via Cheon-`f_n` composition."""

    def __init__(self, degrees: Optional[Tuple[int, ...]] = None) -> None:
        self._degrees: Optional[Tuple[int, ...]] = None
        self._coeffs: Optional[List[np.ndarray]] = None
        if degrees is not None:
            self._set_degrees(degrees)

    def _set_degrees(self, degrees: Tuple[int, ...]) -> None:
        if not degrees:
            raise ValueError("`degrees` must contain at least one polynomial")
        self._degrees = tuple(degrees)
        coeffs: List[np.ndarray] = [_fn_coeffs(d) for d in self._degrees]
        coeffs[-1] = coeffs[-1] / 2.0
        coeffs[-1][0] += 0.5
        self._coeffs = coeffs

    @classmethod
    def from_torch(
        cls,
        module: nn.ReLU,
        input_shape: Tuple[int, ...],
    ) -> Tuple["ReLU", Tuple[int, ...]]:
        return cls(), input_shape

    def __call__(self, x: "EncryptedVector") -> "EncryptedVector":
        if self._coeffs is None:
            raise RuntimeError(
                "ReLU degrees unresolved; call Sequential.compile(context) first or "
                "pass degrees explicitly to ReLU()."
            )
        ctx = x._context
        s = x
        for c in self._coeffs:
            unit = (len(c) - 1 + 3) // 2
            s = ctx._prepare_for(s, unit)
            s = self._horner_odd(s, c)
        x = ctx._prepare_for(x, 1)
        s = ctx._prepare_for(s, 1)
        target = min(x.level, s.level)
        return x.mod_drop_to(target) * s.mod_drop_to(target)

    def mult_depth(self) -> int:
        if self._degrees is None:
            raise RuntimeError("ReLU degrees unresolved; call Sequential.compile(context) first.")
        return sum((d + 3) // 2 for d in self._degrees) + 1

    def forward_plain(self, x: np.ndarray) -> np.ndarray:
        if self._coeffs is None:
            raise RuntimeError("ReLU degrees unresolved; call Sequential.compile(context) first.")
        x = np.asarray(x, dtype=float)
        s = x
        for c in self._coeffs:
            s = np.polynomial.polynomial.polyval(s, c)
        return x * s

    def forward_calibration(self, x: np.ndarray) -> np.ndarray:
        return np.maximum(0.0, np.asarray(x, dtype=float))

    @staticmethod
    def _horner_odd(x: "EncryptedVector", coeffs: np.ndarray) -> "EncryptedVector":
        c0 = float(coeffs[0])
        g_coeffs = coeffs[1::2]
        if len(g_coeffs) == 1:
            out = x * float(g_coeffs[0])
            return out + c0 if c0 != 0.0 else out

        x_sq = x * x
        out = x_sq * float(g_coeffs[-1]) + float(g_coeffs[-2])
        rolling = x_sq.mod_drop_to(out.level)
        for c in reversed(g_coeffs[:-2]):
            out = (out * rolling) + float(c)
            rolling = rolling.mod_drop_to(out.level)
        result = x.mod_drop_to(out.level) * out
        return result + c0 if c0 != 0.0 else result
