from typing import List, Optional

from core.errors import ShapeError
from core.layer import AffineLayer
from core.utils.validate import check_array
from api.tensor import PlaintextTensor


class Linear(AffineLayer):
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
        bias_list: Optional[List[float]] = None
        if bias is not None:
            bias_list = check_array(bias, shape=(out_features,), name="bias").tolist()
        super().__init__(
            in_features, out_features, PlaintextTensor.from_numpy(weight), bias_list
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
