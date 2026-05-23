from math import factorial
from typing import TYPE_CHECKING, List, Tuple

import numpy as np

from core.layer import Layer

if TYPE_CHECKING:
    from api.ciphertext import EncryptedVector


# Default: 12 compositions of f_3 (degree 7). Validated to recover real-ReLU
# accuracy (R^2 ~ 0.79 on California Housing test data).
_DEFAULT_DEGREES: Tuple[int, ...] = (7,) * 12


def _fn_coeffs(degree: int) -> np.ndarray:
    """Coefficients of Cheon et al. (2019) `f_n` polynomial for odd `degree`.

    `f_n` is the unique odd polynomial of degree `2n+1` with f_n(1) = 1 and the
    first n derivatives vanishing at x=1 (by symmetry, the same at x=-1). Its
    composition converges to `sign` on [-1, 1].

    Returns `[c_0, c_1, ..., c_d]` (length `degree + 1`); only odd-indexed
    entries are non-zero.
    """
    if degree < 3 or degree % 2 == 0:
        raise ValueError(f"f_n degree must be odd and >= 3; got {degree}")
    n = (degree - 1) // 2
    # Linear system for odd-indexed coefficients a_1, a_3, ..., a_{2n+1}:
    #   row 0:  sum_k a_{2k+1} = 1                          (p(1) = 1)
    #   row j:  sum_k (2k+1)! / (2k+1-j)! a_{2k+1} = 0      (p^{(j)}(1) = 0)
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
    """Polynomial ReLU via Cheon-`f_n` composition.

    ReLU(x) = x * step(x), where `step(x) = (1 + sign(x)) / 2` is approximated
    by iteratively composing the Cheon-Kim-Kim-Lee-Lee `f_n` polynomials
    (Asiacrypt 2019). Each `f_n` is a closed-form odd polynomial of degree
    `2n+1` whose iteration converges to `sign` on [-1, 1]. The last polynomial
    in the chain is folded into `step` form directly, so `x * step` finishes
    the ReLU without an extra conversion level.

    `degrees`: each entry is one `f_n`'s degree (must be odd >= 3); the length
    is the composition count. Default `(7,) * 12` — 12 copies of f_3 — was
    validated to reach real-ReLU accuracy.

    Input must lie in [-1, 1]; `Sequential.compile(calibration_data=...)`
    folds the surrounding Linear layers so that holds.
    """

    def __init__(self, degrees: Tuple[int, ...] = _DEFAULT_DEGREES) -> None:
        if not degrees:
            raise ValueError("`degrees` must contain at least one polynomial")
        self._degrees = tuple(degrees)
        coeffs: List[np.ndarray] = [_fn_coeffs(d) for d in self._degrees]
        # Convert the last polynomial from sign to step = (1 + sign) / 2:
        # halve every coefficient and add 1/2 to the constant term.
        coeffs[-1] = coeffs[-1] / 2.0
        coeffs[-1][0] += 0.5
        self._coeffs = coeffs

    def __call__(self, x: "EncryptedVector") -> "EncryptedVector":
        # Each f_n composition is one atomic unit; refresh the ciphertext
        # before it if a deep network has run levels low. The original `x`
        # is kept untouched for the final `x * step` multiplication.
        ctx = x._context
        s = x
        for c in self._coeffs:
            unit = (len(c) - 1 + 3) // 2  # (degree + 3) // 2
            s = ctx._prepare_for(s, unit)
            s = self._horner_odd(s, c)
        # s is now step(x); multiply by x. Refresh both first (the ct*ct
        # multiply is itself a one-level unit), then align to a common level —
        # `x` may be above or below `s` depending on where bootstraps fired.
        x = ctx._prepare_for(x, 1)
        s = ctx._prepare_for(s, 1)
        target = min(x.level, s.level)
        return x.mod_drop_to(target) * s.mod_drop_to(target)

    def mult_depth(self) -> int:
        # An odd-degree polynomial via x^2 substitution costs (d+3)/2 levels
        # (1 for x^2 + (d-1)/2 Horner + 1 for the final x * g(x^2)); plus 1
        # for the final x * step at the end.
        return sum((d + 3) // 2 for d in self._degrees) + 1

    def forward_plain(self, x: np.ndarray) -> np.ndarray:
        # Mirror `__call__` exactly: chain the same f_n compositions (with
        # the last one already folded to step form in `__init__`), then
        # multiply by x. So `forward_plain - encrypted_forward` isolates
        # FHE noise from polynomial-approximation error.
        x = np.asarray(x, dtype=float)
        s = x
        for c in self._coeffs:
            s = np.polynomial.polynomial.polyval(s, c)
        return x * s

    def forward_calibration(self, x: np.ndarray) -> np.ndarray:
        # Real ReLU during calibration — the per-neuron folds need to be
        # derived from the original network's activation ranges, not from
        # the polynomial which diverges wildly outside [-1, 1].
        return np.maximum(0.0, np.asarray(x, dtype=float))

    @staticmethod
    def _horner_odd(x: "EncryptedVector", coeffs: np.ndarray) -> "EncryptedVector":
        """Evaluate an odd polynomial (plus optional c_0) at `x`.

        For `p(x) = c_0 + c_1 x + c_3 x^3 + ... + c_d x^d` with even-indexed
        coefficients zero, factor as `p(x) = c_0 + x * g(x^2)` where
        `g(y) = c_1 + c_3 y + c_5 y^2 + ...`. Depth: 1 for x^2, plus a Horner
        of degree `(d-1)/2` in y, plus 1 for the final `x * g(x^2)`.
        """
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
