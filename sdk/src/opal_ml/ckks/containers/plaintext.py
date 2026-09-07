from typing import TYPE_CHECKING, Union

from opal_ml.backend._backend import CKKSPlaintext

if TYPE_CHECKING:
    from opal_ml.ckks.context import FHEContext


class PlaintextVector:
    context: "FHEContext"
    _pt: CKKSPlaintext
    _n_values: int

    def __init__(self, context: "FHEContext", pt: CKKSPlaintext, n_values: int) -> None:
        self.context = context
        self._pt = pt
        self._n_values = n_values

    @property
    def size(self) -> int:
        return self._n_values

    def decode(self) -> list[float]:
        return self.context.decode(self)

    def _values_of(
        self, other: Union["PlaintextVector", list[float], float]
    ) -> list[float]:
        if isinstance(other, PlaintextVector):
            return other.decode()
        if isinstance(other, list):
            return other
        return [float(other)] * self._n_values

    def __add__(
        self, other: Union["PlaintextVector", list[float], float]
    ) -> "PlaintextVector":
        values = self._values_of(other)
        return self.context.encode(
            [a + b for a, b in zip(self.decode(), values, strict=False)]
        )

    def __sub__(
        self, other: Union["PlaintextVector", list[float], float]
    ) -> "PlaintextVector":
        values = self._values_of(other)
        return self.context.encode(
            [a - b for a, b in zip(self.decode(), values, strict=False)]
        )

    def __mul__(
        self, other: Union["PlaintextVector", list[float], float]
    ) -> "PlaintextVector":
        values = self._values_of(other)
        return self.context.encode(
            [a * b for a, b in zip(self.decode(), values, strict=False)]
        )

    def __radd__(
        self, other: Union["PlaintextVector", list[float], float]
    ) -> "PlaintextVector":
        return self.__add__(other)

    def __rsub__(
        self, other: Union["PlaintextVector", list[float], float]
    ) -> "PlaintextVector":
        return (self * -1).__add__(other)

    def __rmul__(
        self, other: Union["PlaintextVector", list[float], float]
    ) -> "PlaintextVector":
        return self.__mul__(other)

    def __repr__(self) -> str:
        return f"PlaintextVector(size={self._n_values}, depth={self._pt.depth})"
