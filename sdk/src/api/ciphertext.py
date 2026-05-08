from typing import TYPE_CHECKING, List, Optional, Union

from core._backend import CKKSCiphertext, CKKSPlaintext
from core.plaintext import PlaintextVector
from api.tensor import PlaintextTensor

if TYPE_CHECKING:
    from api.context import FHEContext


class EncryptedVector:
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

    def copy(self) -> "EncryptedVector":
        return EncryptedVector(self._context, self._ct.copy(), self._n_values)

    def rotate(self, k: int) -> "EncryptedVector":
        return self._context.rotate(self, k)

    def dot(self, weights: List[float]) -> "EncryptedVector":
        if len(weights) != self._n_values:
            raise ValueError(
                f"weights length {len(weights)} != vector size {self._n_values}"
            )
        weighted = self * weights
        summed = weighted._sum_slots(self._n_values)
        return EncryptedVector(self._context, summed._ct, 1)

    def matmul(self, matrix: PlaintextTensor) -> "EncryptedVector":
        if not isinstance(matrix, PlaintextTensor):
            raise TypeError(f"Expected PlaintextTensor, got {type(matrix).__name__}")
        if matrix.ndim != 2:
            raise ValueError(
                f"matmul requires a 2D PlaintextTensor, got {matrix.ndim}D"
            )
        out_features, in_features = matrix.shape
        if in_features != self._n_values:
            raise ValueError(
                f"Matrix columns {in_features} != vector size {self._n_values}"
            )

        n = in_features
        # Pad to next power of 2 so n_padded divides slot_count cleanly,
        # which keeps cyclic rotations consistent with the tile period.
        n_padded = 1 << (n - 1).bit_length() if n > 0 else 1

        # Zero-pad W to n_padded × n_padded.
        W_padded = [
            list(matrix._data[i]) + [0.0] * (n_padded - n) if i < out_features
            else [0.0] * n_padded
            for i in range(n_padded)
        ]

        # Walk r from 0 upwards, advancing `rotated` by a single step each
        # iteration. Galois keys exist for power-of-2 shifts; rotating by 1
        # repeatedly stays within them.
        rotated = self.copy()
        result: Optional[EncryptedVector] = None

        for r in range(n_padded):
            diag_r = [W_padded[i][(i + r) % n_padded] for i in range(n_padded)]
            if not all(v == 0.0 for v in diag_r):
                pt = self._encode_and_align(diag_r)
                term = rotated.copy()
                self._context._ops.multiply_plain_inplace(term._ct, pt)
                self._context._ops.rescale_inplace(term._ct)
                result = term if result is None else result + term
            if r < n_padded - 1:
                rotated = self._context.rotate(rotated, 1)

        if result is None:
            raise ValueError("All matrix diagonals are zero")
        return EncryptedVector(self._context, result._ct, out_features)

    def __add__(
        self, other: Union["EncryptedVector", PlaintextVector, List[float], float]
    ) -> "EncryptedVector":
        res = self.copy()
        if isinstance(other, EncryptedVector):
            self._context._ops.add_inplace(res._ct, other._ct.copy())
        elif isinstance(other, PlaintextVector):
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

    def __sub__(
        self, other: Union["EncryptedVector", PlaintextVector, List[float], float]
    ) -> "EncryptedVector":
        res = self.copy()
        if isinstance(other, EncryptedVector):
            self._context._ops.sub_inplace(res._ct, other._ct.copy())
        elif isinstance(other, PlaintextVector):
            if other._pt.depth != res._ct.depth:
                raise ValueError(
                    f"Depth mismatch: ciphertext depth={res._ct.depth}, "
                    f"plaintext depth={other._pt.depth}."
                )
            self._context._ops.sub_plain_inplace(res._ct, other._pt)
        else:
            self._context._ops.sub_plain_inplace(res._ct, self._encode_and_align(other))
        return res

    def __mul__(
        self, other: Union["EncryptedVector", PlaintextVector, List[float], float]
    ) -> "EncryptedVector":
        res = self.copy()
        if isinstance(other, EncryptedVector):
            self._context._ops.multiply_inplace(res._ct, other._ct.copy())
            self._context._ops.relinearize_inplace(res._ct, self._context._rk)
            self._context._ops.rescale_inplace(res._ct)
        elif isinstance(other, PlaintextVector):
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

    def __radd__(
        self, other: Union["EncryptedVector", PlaintextVector, List[float], float]
    ) -> "EncryptedVector":
        return self.__add__(other)

    def __rsub__(
        self, other: Union["EncryptedVector", PlaintextVector, List[float], float]
    ) -> "EncryptedVector":
        return (self * -1).__add__(other)

    def __rmul__(
        self, other: Union["EncryptedVector", PlaintextVector, List[float], float]
    ) -> "EncryptedVector":
        return self.__mul__(other)

    def _encode_and_align(self, values: Union[List[float], float]) -> CKKSPlaintext:
        if isinstance(values, (int, float)):
            values = [float(values)] * self._n_values
        pt = self._context.encode(values)
        while pt._pt.depth < self._ct.depth:
            self._context._ops.mod_drop_plain_inplace(pt._pt)
        return pt._pt

    def _sum_slots(self, n: int) -> "EncryptedVector":
        result = self.copy()
        step = 1
        while step < n:
            rotated = self._context.rotate(result, step)
            result = result + rotated
            step *= 2
        return result

    def _replicate_slot0(self) -> "EncryptedVector":
        slot_count = self._context._poly_modulus_degree // 2
        result = self.copy()
        step = slot_count // 2
        while step >= 1:
            result = result + self._context.rotate(result, step)
            step //= 2
        return result
