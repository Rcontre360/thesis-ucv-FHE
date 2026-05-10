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
        """Plaintext-matrix × ciphertext-vector via cyclic-wrap diagonals.

        Computes y = W @ x for our `W ∈ ℝ^{out×in}` convention and encrypted
        x of size `in`. The algorithm is the rectangular Halevi–Shoup variant
        used by TenSEAL: diagonals walk through `M = Wᵀ` (shape `(in, out)`)
        with cyclic indexing in both dimensions.

        Slot pattern invariant: input ciphertext carries `x` replicated to
        slot_count (`enc_x[k] = x[k mod in]`). Output is replicated with
        period `out_features` (`result[k] = y[k mod out]`). No zero padding,
        no tile-period bookkeeping.
        """
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

        # In TenSEAL's convention M is (n_rows, n_cols) with n_rows = enc.size().
        # Our W is (out, in), so we read it as if it were Wᵀ: M[r][c] = W[c][r].
        n_rows = in_features
        n_cols = out_features
        slot_count = self._context._poly_modulus_degree // 2
        diag_len = min(slot_count, n_rows * n_cols)

        # Formulation B: rotate the ciphertext by 1 incrementally. The temp ct at
        # iteration `local_i` is rotate(enc_x, local_i) — uses only rotate-by-1
        # Galois keys, identical result to TenSEAL's rotate-after-multiply form.
        rotated = self.copy()
        result: Optional[EncryptedVector] = None

        for local_i in range(n_rows):
            # diag_local_i[k] = M[(local_i + k) mod n_rows][k mod n_cols]
            #                 = W[k mod out_features][(local_i + k) mod in_features]
            diag = [
                matrix._data[k % n_cols][(local_i + k) % n_rows]
                for k in range(diag_len)
            ]
            if not all(v == 0.0 for v in diag):
                pt = self._encode_and_align(diag)
                term = rotated.copy()
                self._context._ops.multiply_plain_inplace(term._ct, pt)
                self._context._ops.rescale_inplace(term._ct)
                result = term if result is None else result + term
            if local_i < n_rows - 1:
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
            values_list: List[float] = [float(values)] * self._n_values
        else:
            values_list = list(values)
        # encode replicates to slot_count, so the resulting plaintext aligns
        # slot-by-slot with replicated ciphertexts regardless of len(values).
        pt = self._context.encode(values_list)
        while pt._pt.depth < self._ct.depth:
            self._context._ops.mod_drop_plain_inplace(pt._pt)
        return pt._pt

    def _sum_slots(self, n: int) -> "EncryptedVector":
        """Sum the first n slots into slot 0.

        Recursive power-of-2 decomposition (à la TenSEAL's `sum_vector`).
        For a power-of-2 `n` this collapses to the textbook `log2(n)` doubling
        sum. For non-power-of-2 `n`, it splits `n = bp2 + (n − bp2)` where
        `bp2` is the largest power of 2 ≤ n, doubling-sums the head, recurses
        on the rotated tail, and adds them. Correct for replicated inputs at
        any `n ≥ 1`, no extra multiplicative-level cost.
        """
        if n <= 1:
            return self.copy()

        bp2 = 1 << (n.bit_length() - 1)
        if bp2 == n:
            # n is a power of 2 — pure doubling sum.
            result = self.copy()
            step = bp2 // 2
            while step >= 1:
                result = result + self._context.rotate(result, step)
                step //= 2
            return result

        # bp2 < n: rotate the original by bp2 to expose the tail at slot 0,
        # recurse on the tail, doubling-sum the head, and combine.
        rest = self._context.rotate(self, bp2)._sum_slots(n - bp2)

        result = self.copy()
        step = bp2 // 2
        while step >= 1:
            result = result + self._context.rotate(result, step)
            step //= 2

        return result + rest
