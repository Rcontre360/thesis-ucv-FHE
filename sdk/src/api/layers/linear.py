from typing import TYPE_CHECKING, List, Optional

from api._validate import check_array
from api.errors import ShapeError
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
        weight: object,
        bias: Optional[object] = None,
    ) -> None:
        weight = check_array(weight, shape=(out_features, in_features), name="weight")
        if bias is not None:
            bias = check_array(bias, shape=(out_features,), name="bias")

        self.in_features = in_features
        self.out_features = out_features
        self._weight = PlaintextTensor.from_numpy(weight)
        self._bias: Optional[List[float]] = (
            bias.tolist() if bias is not None else None
        )

    def prepare_input(self, raw_data: object) -> List[float]:
        """Validate a flat 1-D list/tuple/numpy array of length `in_features`."""
        arr = check_array(raw_data, name="Linear input")
        if arr.ndim != 1:
            raise ShapeError(
                f"Linear input must be 1-D, got {arr.ndim}-D — "
                "did you mean to use Conv2D as the first layer?"
            )
        if arr.size != self.in_features:
            raise ShapeError(
                f"input size {arr.size} != in_features {self.in_features}"
            )
        return arr.tolist()

    def __call__(self, x: "EncryptedVector") -> "EncryptedVector":
        if x.size != self.in_features:
            raise ShapeError(
                f"input size {x.size} != in_features {self.in_features}"
            )
        out = x.matmul(self._weight)
        if self._bias is not None:
            out = out + self._bias
        return out
