from typing import TYPE_CHECKING, List, Union

from fhe_sdk._backend import CKKSCiphertext, CKKSPlaintext
from fhe_sdk.plaintext import Plaintext

if TYPE_CHECKING:
    from fhe_sdk.context import FHEContext


class Ciphertext:
    _context: "FHEContext"
    _ct: CKKSCiphertext
    _n_values: int

    def __init__(self, context: "FHEContext", ct: CKKSCiphertext, n_values: int) -> None:
        self._context = context
        self._ct = ct
        self._n_values = n_values

    @property
    def size(self) -> int:
        return self._n_values

    def decrypt(self) -> List[float]:
        return self._context.decrypt(self)

    def copy(self) -> "Ciphertext":
        return Ciphertext(self._context, self._ct.copy(), self._n_values)

    def _encode_and_align(self, values: Union[List[float], float]) -> CKKSPlaintext:
        """Encode a list or scalar and mod-drop to match this ciphertext's depth."""
        if isinstance(values, (int, float)):
            values = [float(values)] * self._n_values
        pt = self._context.encode(values)
        while pt._pt.depth < self._ct.depth:
            self._context._ops.mod_drop_plain_inplace(pt._pt)
        return pt._pt

    def __add__(self, other: Union["Ciphertext", Plaintext, List[float], float]) -> "Ciphertext":
        res = self.copy()

        if isinstance(other, Ciphertext):
            self._context._ops.add_inplace(res._ct, other._ct.copy())
        elif isinstance(other, Plaintext):
            if other._pt.depth != res._ct.depth:
                raise ValueError(
                    f"Depth mismatch: ciphertext depth={res._ct.depth}, "
                    f"plaintext depth={other._pt.depth}. "
                    "Encode the plaintext at the matching depth or pass a list/scalar."
                )
            self._context._ops.add_plain_inplace(res._ct, other._pt)
        else:
            self._context._ops.add_plain_inplace(res._ct, self._encode_and_align(other))

        return res

    def __sub__(self, other: Union["Ciphertext", Plaintext, List[float], float]) -> "Ciphertext":
        res = self.copy()

        if isinstance(other, Ciphertext):
            self._context._ops.sub_inplace(res._ct, other._ct.copy())
        elif isinstance(other, Plaintext):
            if other._pt.depth != res._ct.depth:
                raise ValueError(
                    f"Depth mismatch: ciphertext depth={res._ct.depth}, "
                    f"plaintext depth={other._pt.depth}."
                )
            self._context._ops.sub_plain_inplace(res._ct, other._pt)
        else:
            self._context._ops.sub_plain_inplace(res._ct, self._encode_and_align(other))

        return res

    def __mul__(self, other: Union["Ciphertext", Plaintext, List[float], float]) -> "Ciphertext":
        res = self.copy()

        if isinstance(other, Ciphertext):
            self._context._ops.multiply_inplace(res._ct, other._ct.copy())
            self._context._ops.relinearize_inplace(res._ct, self._context._rk)
            self._context._ops.rescale_inplace(res._ct)
        elif isinstance(other, Plaintext):
            if other._pt.depth != res._ct.depth:
                raise ValueError(
                    f"Depth mismatch: ciphertext depth={res._ct.depth}, "
                    f"plaintext depth={other._pt.depth}."
                )
            self._context._ops.multiply_plain_inplace(res._ct, other._pt)
            self._context._ops.rescale_inplace(res._ct)
        else:
            self._context._ops.multiply_plain_inplace(res._ct, self._encode_and_align(other))
            self._context._ops.rescale_inplace(res._ct)

        return res

    def __radd__(self, other: Union["Ciphertext", Plaintext, List[float], float]) -> "Ciphertext":
        return self.__add__(other)

    def __rsub__(self, other: Union["Ciphertext", Plaintext, List[float], float]) -> "Ciphertext":
        return (self * -1).__add__(other)

    def __rmul__(self, other: Union["Ciphertext", Plaintext, List[float], float]) -> "Ciphertext":
        return self.__mul__(other)
