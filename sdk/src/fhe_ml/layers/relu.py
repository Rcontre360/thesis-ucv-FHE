from typing import List, Optional, Tuple

import numpy as np
from numpy.polynomial import Polynomial
import torch.nn as nn

from fhe_ml.ckks.containers.ciphertext import EncryptedVector
from fhe_ml.layers.base import Layer


def _fn_coeffs(degree: int) -> np.ndarray:
    """Cheon `f_n` of degree 2n+1: unique odd poly with f(1)=1 and f^(j)(1)=0 for j=1..n (ePrint 2019/417)."""
    if degree < 3 or degree % 2 == 0:
        raise ValueError(f"f_n degree must be odd and >= 3; got {degree}")
    n = (degree - 1) // 2
    # Basis: odd monomials b_k(x) = x^(2k+1) for k=0..n.
    odd_monomials = [Polynomial.basis(2 * k + 1) for k in range(n + 1)]
    # Row j is the j-th derivative of each basis monomial evaluated at x=1.
    constraint_matrix = np.array(
        [[m.deriv(j)(1.0) for m in odd_monomials] for j in range(n + 1)]
    )
    # Right-hand side encodes f(1)=1 and f^(j)(1)=0 for j>=1.
    rhs = np.zeros(n + 1)
    rhs[0] = 1.0
    odd_coeffs = np.linalg.solve(constraint_matrix, rhs)
    coeffs = np.zeros(degree + 1)
    coeffs[1::2] = odd_coeffs
    return coeffs


class ReLU(Layer):
    """Polynomial ReLU via Cheon-`f_n` composition."""

    def __init__(self) -> None:
        self._degrees: Optional[Tuple[int, ...]] = None
        self._coeffs: Optional[List[np.ndarray]] = None

    def set_degrees(self, degrees: Tuple[int, ...]) -> None:
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

    def __call__(self, x: EncryptedVector) -> EncryptedVector:
        if self._coeffs is None:
            raise RuntimeError(
                "ReLU degrees unresolved; call Sequential.compile(context) "
                "or ReLU.set_degrees(...) first."
            )
        ctx = x._context
        # Composed f_n chain pushes inputs toward the sign of x (±1 in the working range).
        sign_approx = x
        for poly_coeffs in self._coeffs:
            # Each f_n of degree d burns (d+3)/2 levels: 1 for x², (d-3)/2 Horner mults, 1 final x·q(x²).
            levels_needed = (len(poly_coeffs) - 1 + 3) // 2
            sign_approx = ctx._prepare_for(sign_approx, levels_needed)
            sign_approx = self._eval_odd_poly(sign_approx, poly_coeffs)
        # Smooth ReLU(x) = x · (sign(x)+1)/2; the +0.5 was folded into the last poly's constant.
        x = ctx._prepare_for(x, 1)
        sign_approx = ctx._prepare_for(sign_approx, 1)
        target_level = min(x.level, sign_approx.level)
        return x.mod_drop_to(target_level) * sign_approx.mod_drop_to(target_level)

    def mult_depth(self) -> int:
        if self._degrees is None:
            raise RuntimeError("ReLU degrees unresolved; call Sequential.compile(context) or ReLU.set_degrees(...) first.")
        return sum((d + 3) // 2 for d in self._degrees) + 1

    def forward_plain(self, x: np.ndarray) -> np.ndarray:
        if self._coeffs is None:
            raise RuntimeError("ReLU degrees unresolved; call Sequential.compile(context) or ReLU.set_degrees(...) first.")
        x = np.asarray(x, dtype=float)
        s = x
        for c in self._coeffs:
            s = np.polynomial.polynomial.polyval(s, c)
        return x * s

    def forward_calibration(self, x: np.ndarray) -> np.ndarray:
        return np.maximum(0.0, np.asarray(x, dtype=float))

    @staticmethod
    def _eval_odd_poly(x: EncryptedVector, coeffs: np.ndarray) -> EncryptedVector:
        """Evaluate p(x) = c0 + x · q(x²), where q's coefficients are the odd-indexed entries of `coeffs`."""
        # c0 is 0 for inner f_n composition steps; only the final smooth-ReLU layer carries 0.5.
        constant_term = float(coeffs[0])
        # p(x) = c0 + sum_k coeffs[2k+1] · x^(2k+1) = c0 + x · sum_k coeffs[2k+1] · (x²)^k.
        q_coeffs = coeffs[1::2]

        # Degenerate case: q is a single coefficient, so p collapses to a linear map.
        if len(q_coeffs) == 1:
            result = x * float(q_coeffs[0])
            return result + constant_term if constant_term != 0.0 else result

        q_at_x_sq = ReLU._horner_on_x_squared(x, q_coeffs)
        # x has not been touched since entry; drop it to align with q(x²)'s consumed levels.
        result = x.mod_drop_to(q_at_x_sq.level) * q_at_x_sq
        return result + constant_term if constant_term != 0.0 else result

    @staticmethod
    def _horner_on_x_squared(
        x: EncryptedVector, q_coeffs: np.ndarray
    ) -> EncryptedVector:
        """Compute q(x²) via Horner. Saves depth vs. Horner on x: needs deg(q) mults, not 2·deg(q)."""
        # u := x² uses one ciphertext-ciphertext multiplication and one rescale.
        x_sq = x * x
        # Seed Horner with the top two coefficients: out = q_n · u + q_{n-1}.
        out = x_sq * float(q_coeffs[-1]) + float(q_coeffs[-2])
        # `rolling_x_sq` is u kept level-aligned with `out` so the next mult is legal.
        rolling_x_sq = x_sq.mod_drop_to(out.level)
        # Inner Horner: out := out · u + q_k, for k = n-2 down to 0.
        for q_k in reversed(q_coeffs[:-2]):
            out = out * rolling_x_sq + float(q_k)
            rolling_x_sq = rolling_x_sq.mod_drop_to(out.level)
        return out
