from typing import TYPE_CHECKING, List, Union

from fhe_sdk._backend import CKKSPlaintext

if TYPE_CHECKING:
    from fhe_sdk.context import FHEContext


class Plaintext:
    _context: "FHEContext"
    _pt: CKKSPlaintext
    _n_values: int

    def __init__(self, context: "FHEContext", pt: CKKSPlaintext, n_values: int) -> None:
        self._context = context
        self._pt = pt
        self._n_values = n_values

    @property
    def size(self) -> int:
        return self._n_values

    def decode(self) -> List[float]:
        return self._context.decode(self)

    def __add__(self, other: Union["Plaintext", List[float], float]) -> "Plaintext":
        values = other if isinstance(other, list) else [float(other)] * self._n_values
        result = [a + b for a, b in zip(self.decode(), values)]
        return self._context.encode(result)

    def __sub__(self, other: Union["Plaintext", List[float], float]) -> "Plaintext":
        values = other if isinstance(other, list) else [float(other)] * self._n_values
        result = [a - b for a, b in zip(self.decode(), values)]
        return self._context.encode(result)

    def __mul__(self, other: Union["Plaintext", List[float], float]) -> "Plaintext":
        values = other if isinstance(other, list) else [float(other)] * self._n_values
        result = [a * b for a, b in zip(self.decode(), values)]
        return self._context.encode(result)

    def __radd__(self, other: Union["Plaintext", List[float], float]) -> "Plaintext":
        return self.__add__(other)

    def __rsub__(self, other: Union["Plaintext", List[float], float]) -> "Plaintext":
        return (self * -1).__add__(other)

    def __rmul__(self, other: Union["Plaintext", List[float], float]) -> "Plaintext":
        return self.__mul__(other)

    def __repr__(self) -> str:
        return f"Plaintext(size={self._n_values}, depth={self._pt.depth})"
