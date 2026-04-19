from typing import TYPE_CHECKING, List, Optional

from api.tensor import PlaintextTensor

if TYPE_CHECKING:
    from api.ciphertext import EncryptedVector


class Linear:
    """Fully-connected layer for encrypted vectors.

    Applies y = x @ W^T + b where W is a plaintext weight matrix and b is an
    optional plaintext bias vector.  Wraps EncryptedVector.matmul(), which uses
    the diagonal BSGS algorithm and consumes no extra multiplication levels.

    Args:
        in_features:  number of input elements per sample.
        out_features: number of output elements per sample.
        weight:       2-D list of shape (out_features, in_features).
        bias:         1-D list of length out_features, or None.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        weight: List[List[float]],
        bias: Optional[List[float]] = None,
    ) -> None:
        if len(weight) != out_features:
            raise ValueError(
                f"weight must have {out_features} rows, got {len(weight)}"
            )
        if any(len(row) != in_features for row in weight):
            raise ValueError(
                f"every weight row must have {in_features} columns"
            )
        if bias is not None and len(bias) != out_features:
            raise ValueError(
                f"bias length {len(bias)} != out_features {out_features}"
            )
        self.in_features = in_features
        self.out_features = out_features
        self._weight = PlaintextTensor(weight)
        self._bias: Optional[List[float]] = list(bias) if bias is not None else None

    def __call__(self, x: "EncryptedVector") -> "EncryptedVector":
        if x.size != self.in_features:
            raise ValueError(
                f"input size {x.size} != in_features {self.in_features}"
            )
        out = x.matmul(self._weight)
        if self._bias is not None:
            out = out + self._bias
        return out
