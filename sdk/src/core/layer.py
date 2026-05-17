from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, List, Optional

import numpy as np

from core.errors import ShapeError

if TYPE_CHECKING:
    from api.ciphertext import EncryptedVector
    from api.tensor import PlaintextTensor


class Layer(ABC):
    """Interface for `Sequential` elements: a callable EncryptedVector -> EncryptedVector.

    Subclasses must implement `__call__`. Layers usable as a model's first
    element override `prepare_input`; others inherit the raising default.
    """

    @abstractmethod
    def __call__(self, x: "EncryptedVector") -> "EncryptedVector":
        ...

    @abstractmethod
    def mult_depth(self) -> int:
        """Multiplicative levels this layer consumes (critical path, not count).

        `Sequential.compile` sums these to size the bootstrapping schedule.
        """
        ...

    @abstractmethod
    def forward_plain(self, x: np.ndarray) -> np.ndarray:
        """Plaintext numpy forward pass — for calibration and as a reference.

        Accepts a single sample `(features,)` or a batch `(N, features)`.
        """
        ...

    def prepare_input(self, raw_data: object) -> List[float]:
        raise NotImplementedError(
            f"{type(self).__name__} cannot be a model's input layer; "
            "use a weighted layer (Linear, Conv2D, ...) as the first element."
        )


class AffineLayer(Layer):
    """Base for layers computing y = W @ x + b with plaintext W and b.

    Subclasses build the weight matrix and bias in their own `__init__`, then
    call `super().__init__(...)`; the forward pass is shared here.
    """

    in_features: int
    out_features: int
    _weight: "PlaintextTensor"
    _bias: Optional[List[float]]

    def __init__(
        self,
        in_features: int,
        out_features: int,
        weight: "PlaintextTensor",
        bias: Optional[List[float]],
    ) -> None:
        self.in_features = in_features
        self.out_features = out_features
        self._weight = weight
        self._bias = bias

    def __call__(self, x: "EncryptedVector") -> "EncryptedVector":
        if x.size != self.in_features:
            raise ShapeError(
                f"input size {x.size} != in_features {self.in_features}"
            )
        out = x.matmul(self._weight)
        if self._bias is not None:
            out = out + self._bias
        return out

    def mult_depth(self) -> int:
        return 1

    def forward_plain(self, x: np.ndarray) -> np.ndarray:
        out = np.asarray(x, dtype=float) @ self._weight.to_numpy().T
        if self._bias is not None:
            out = out + np.asarray(self._bias, dtype=float)
        return out
