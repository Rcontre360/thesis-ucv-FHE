from typing import TYPE_CHECKING, List, Optional, Union

from core._backend import CKKSCiphertext, CKKSPlaintext
from core.plaintext import PlaintextVector
from api.tensor import PlaintextTensor

if TYPE_CHECKING:
    from api.context import FHEContext


def _next_pow2(n: int) -> int:
    return 1 << (n - 1).bit_length() if n > 0 else 1


class EncryptedVector:
    _context: "FHEContext"
    _ct: CKKSCiphertext
    _n_values: int
    _period: int

    def __init__(
        self,
        context: "FHEContext",
        ct: CKKSCiphertext,
        n_values: int,
        period: Optional[int] = None,
    ) -> None:
        self._context = context
        self._ct = ct
        self._n_values = n_values
        self._period = period if period is not None else _next_pow2(n_values)

    @property
    def size(self) -> int:
        return self._n_values

    @property
    def period(self) -> int:
        return self._period

    def decrypt(self) -> List[float]:
        return self._context.decrypt(self)

    def copy(self) -> "EncryptedVector":
        return EncryptedVector(self._context, self._ct.copy(), self._n_values, self._period)

    def rotate(self, k: int) -> "EncryptedVector":
        return self._context.rotate(self, k)

    def dot(self, weights: List[float]) -> "EncryptedVector":
        if len(weights) != self._n_values:
            raise ValueError(
                f"weights length {len(weights)} != vector size {self._n_values}"
            )
        weighted = self * weights
        summed = weighted._sum_slots(self._n_values)
        return EncryptedVector(self._context, summed._ct, 1, self._period)

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

        n_padded = _next_pow2(in_features)
        m_padded = _next_pow2(out_features)
        # Rectangular Halevi–Shoup: run the diagonal algorithm on a square
        # s × s zero-padded view of W. s must fit both dims AND not shrink the
        # ciphertext's tile period (CKKS rotations preserve whatever period the
        # ciphertext already has; shrinking would read zeros from previous-layer
        # padding instead of the wrap-around values the algorithm assumes).
        s = max(n_padded, m_padded, self._period)

        # Lift the ciphertext's tile period to s when the algorithm needs more
        # working slots than the ciphertext currently exposes. Costs 1 mul level.
        x = self if self._period >= s else self._extend_period(s)

        # Zero-pad W to s × s.
        W_padded = []
        for i in range(s):
            if i < out_features:
                row = list(matrix._data[i]) + [0.0] * (s - in_features)
            else:
                row = [0.0] * s
            W_padded.append(row)

        # Walk r from 0 upwards, advancing `rotated` by a single step each
        # iteration. Galois keys exist for power-of-2 shifts; rotating by 1
        # repeatedly stays within them.
        rotated = x.copy()
        result: Optional[EncryptedVector] = None

        for r in range(s):
            diag_r = [W_padded[i][(i + r) % s] for i in range(s)]
            if not all(v == 0.0 for v in diag_r):
                pt = x._encode_and_align(diag_r)
                term = rotated.copy()
                self._context._ops.multiply_plain_inplace(term._ct, pt)
                self._context._ops.rescale_inplace(term._ct)
                result = term if result is None else result + term
            if r < s - 1:
                rotated = self._context.rotate(rotated, 1)

        if result is None:
            raise ValueError("All matrix diagonals are zero")
        return EncryptedVector(self._context, result._ct, out_features, period=s)

    def _extend_period(self, new_period: int) -> "EncryptedVector":
        """Mask out tile copies so the ciphertext's tile period grows to new_period.

        The current tile is [v_0, …, v_{n-1}, 0, …, 0] of length self._period,
        repeated across slot_count. After masking it becomes the same prefix
        followed by zeros up to new_period, repeated. Costs 1 multiplicative
        level (multiply_plain + rescale).
        """
        if new_period == self._period:
            return self
        if new_period < self._period:
            raise ValueError(
                f"Cannot shrink period from {self._period} to {new_period}"
            )
        if new_period & (new_period - 1) != 0:
            raise ValueError(f"new_period must be a power of two, got {new_period}")

        mask = [1.0] * self._period + [0.0] * (new_period - self._period)
        pt = self._context.encode(mask)
        new_ct = self._ct.copy()
        while pt._pt.depth < new_ct.depth:
            self._context._ops.mod_drop_plain_inplace(pt._pt)
        self._context._ops.multiply_plain_inplace(new_ct, pt._pt)
        self._context._ops.rescale_inplace(new_ct)
        return EncryptedVector(self._context, new_ct, self._n_values, period=new_period)

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
