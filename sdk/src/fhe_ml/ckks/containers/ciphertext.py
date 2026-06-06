from typing import TYPE_CHECKING, List, Optional, Union

import numpy as np

from fhe_ml.backend._backend import CKKSCiphertext, CKKSPlaintext
from fhe_ml.utils.errors import ShapeError
from fhe_ml.ckks.containers.plaintext import PlaintextVector
from fhe_ml.ckks.containers.tensor import PlaintextTensor

if TYPE_CHECKING:
    from fhe_ml.ckks.context import FHEContext


class EncryptedVector:
    _context: "FHEContext"
    _ct: CKKSCiphertext
    _n_values: int

    def __init__(
        self,
        context: "FHEContext",
        ct: CKKSCiphertext,
        n_values: int,
    ) -> None:
        self._context = context
        self._ct = ct
        self._n_values = n_values

    @property
    def size(self) -> int:
        return self._n_values

    @property
    def level(self) -> int:
        """Remaining usable multiplication levels in this ciphertext."""
        return self._ct.level

    def decrypt(self) -> List[float]:
        return self._context.decrypt(self)

    def copy(self) -> "EncryptedVector":
        return EncryptedVector(self._context, self._ct.copy(), self._n_values)

    def mod_drop_to(self, target_level: int) -> "EncryptedVector":
        """Drop modulus primes until `self.level == target_level` (no-op if already)."""
        res = self.copy()
        while res._ct.level > target_level:
            self._context._ops.mod_drop_inplace(res._ct)
        return res

    def rotate(self, k: int) -> "EncryptedVector":
        return self._context.rotate(self, k)

    def dot(self, weights: List[float]) -> "EncryptedVector":
        if len(weights) != self._n_values:
            raise ShapeError(
                f"weights length {len(weights)} != vector size {self._n_values}"
            )
        weighted = self * weights
        summed = weighted._sum_slots(self._n_values)
        return EncryptedVector(self._context, summed._ct, 1)

    def matmul(self, matrix: PlaintextTensor) -> "EncryptedVector":
        """y = W @ x via the rectangular Halevi-Shoup cyclic-wrap diagonal method.

        Input and output ciphertexts are replicated to slot_count
        (enc_x[k] = x[k mod in], result[k] = y[k mod out]).
        Requires the matrix to have been pre-encoded via PlaintextTensor.encode
        (called automatically by Sequential.compile).
        """
        if not isinstance(matrix, PlaintextTensor):
            raise TypeError(f"Expected PlaintextTensor, got {type(matrix).__name__}")
        if matrix.ndim != 2:
            raise ShapeError(
                f"matmul requires a 2D PlaintextTensor, got {matrix.ndim}D"
            )
        if matrix._encoded_diagonals is None:
            raise RuntimeError(
                "PlaintextTensor has not been encoded. Call Sequential.compile(context) "
                "before inference, or PlaintextTensor.encode(context) for standalone use."
            )
        out_features, in_features = matrix.shape
        if in_features != self._n_values:
            raise ShapeError(
                f"Matrix columns {in_features} != vector size {self._n_values}"
            )

        n_rows = in_features
        n_cols = out_features
        encoded_diagonals = matrix._encoded_diagonals

        rotated = self.copy()
        result: Optional[EncryptedVector] = None

        for local_i in range(n_rows):
            stored_pt = encoded_diagonals[local_i]
            if stored_pt is not None:
                pt = stored_pt.copy()
                while pt.depth < self._ct.depth:
                    self._context._ops.mod_drop_plain_inplace(pt)
                term = rotated.copy()
                self._context._ops.multiply_plain_inplace(term._ct, pt)
                self._context._ops.rescale_inplace(term._ct)
                result = term if result is None else result + term
            if local_i < n_rows - 1:
                rotated = self._context.rotate(rotated, 1)

        if result is None:
            raise ShapeError("All matrix diagonals are zero")
        return EncryptedVector(self._context, result._ct, out_features)

    def __add__(
        self, other: Union["EncryptedVector", PlaintextVector, List[float], float]
    ) -> "EncryptedVector":
        res = self.copy()
        if isinstance(other, EncryptedVector):
            self._context._ops.add_inplace(res._ct, other._ct.copy())
        else:
            self._context._ops.add_plain_inplace(res._ct, self._resolve_plain(other))
        return res

    def __sub__(
        self, other: Union["EncryptedVector", PlaintextVector, List[float], float]
    ) -> "EncryptedVector":
        res = self.copy()
        if isinstance(other, EncryptedVector):
            self._context._ops.sub_inplace(res._ct, other._ct.copy())
        else:
            self._context._ops.sub_plain_inplace(res._ct, self._resolve_plain(other))
        return res

    def __mul__(
        self, other: Union["EncryptedVector", PlaintextVector, List[float], float]
    ) -> "EncryptedVector":
        res = self.copy()
        if isinstance(other, EncryptedVector):
            self._context._ops.multiply_inplace(res._ct, other._ct.copy())
            self._context._ops.relinearize_inplace(res._ct, self._context._rk)
        else:
            self._context._ops.multiply_plain_inplace(res._ct, self._resolve_plain(other))
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

    def _resolve_plain(
        self, other: Union[PlaintextVector, List[float], float]
    ) -> CKKSPlaintext:
        """Resolve a non-ciphertext operand to a depth-aligned CKKSPlaintext."""
        if isinstance(other, PlaintextVector):
            if other._pt.depth != self._ct.depth:
                raise ShapeError(
                    f"Depth mismatch: ciphertext depth={self._ct.depth}, "
                    f"plaintext depth={other._pt.depth}. "
                    "Encode the plaintext at the matching depth or pass a list/scalar."
                )
            return other._pt
        return self._encode_and_align(other)

    def _encode_and_align(self, values: Union[List[float], float]) -> CKKSPlaintext:
        if isinstance(values, (int, float)):
            values_list: List[float] = [float(values)] * self._n_values
        else:
            values_list = list(values)
        pt = self._context.encode(values_list)
        while pt._pt.depth < self._ct.depth:
            self._context._ops.mod_drop_plain_inplace(pt._pt)
        return pt._pt

    def _sum_slots(self, n: int) -> "EncryptedVector":
        """Sum the first n slots into slot 0 via power-of-2 decomposition.

        Splits a non-power-of-2 `n` into a doubling-summed head plus a
        recursive tail. Correct for replicated inputs at any n >= 1.
        """
        if n <= 1:
            return self.copy()

        bp2 = 1 << (n.bit_length() - 1)
        if bp2 == n:
            result = self.copy()
            step = bp2 // 2
            while step >= 1:
                result = result + self._context.rotate(result, step)
                step //= 2
            return result

        rest = self._context.rotate(self, bp2)._sum_slots(n - bp2)

        result = self.copy()
        step = bp2 // 2
        while step >= 1:
            result = result + self._context.rotate(result, step)
            step //= 2

        return result + rest
