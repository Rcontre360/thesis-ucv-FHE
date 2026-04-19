from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from api.ciphertext import EncryptedVector


class ReLU:
    """Approximate ReLU via degree-2 polynomial: 0.125x² + 0.5x + 0.375.

    Least-squares Chebyshev fit over [-1, 1]. Consumes 1 multiplication level.
    """

    def __call__(self, x: "EncryptedVector") -> "EncryptedVector":
        # Factor as (x² + 4x) * 0.125 + 0.375 so both operands of the
        # final ct+ct addition are at the same depth.
        x_sq = x * x       # depth+1  (ct*ct + relin + rescale)
        x_4  = x * 4.0     # depth+1  (ct*pt + rescale)
        return (x_sq + x_4) * 0.125 + 0.375
