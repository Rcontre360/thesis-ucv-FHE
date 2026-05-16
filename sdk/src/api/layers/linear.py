from typing import TYPE_CHECKING, List, Optional

from api.tensor import PlaintextTensor
from api.layers.base import Layer

if TYPE_CHECKING:
    from api.ciphertext import EncryptedVector


class Linear(Layer):
    """Fully-connected layer: y = W @ x + b, no extra multiplication depth.

    weight has shape (out_features, in_features); bias has length out_features
    or is None.
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

    def prepare_input(self, raw_data: object) -> List[float]:
        """Validate a flat 1-D list/tuple/numpy array of length `in_features`."""
        if hasattr(raw_data, "tolist"):
            raw_data = raw_data.tolist()
        if not isinstance(raw_data, (list, tuple)):
            raise TypeError(
                f"Linear.prepare_input expects a 1-D list, got {type(raw_data).__name__}"
            )
        flat = list(raw_data)
        if any(isinstance(v, (list, tuple)) for v in flat):
            raise ValueError(
                "Linear.prepare_input expects a flat 1-D sequence; "
                "got nested data — did you mean to use Conv2D as the first layer?"
            )
        if len(flat) != self.in_features:
            raise ValueError(
                f"input size {len(flat)} != in_features {self.in_features}"
            )
        return [float(v) for v in flat]

    def __call__(self, x: "EncryptedVector") -> "EncryptedVector":
        if x.size != self.in_features:
            raise ValueError(
                f"input size {x.size} != in_features {self.in_features}"
            )
        out = x.matmul(self._weight)
        if self._bias is not None:
            out = out + self._bias
        return out
