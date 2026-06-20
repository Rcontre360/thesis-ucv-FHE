from typing import TYPE_CHECKING, Union

from fhe_ml.backend._backend import CKKSCiphertext, CKKSPlaintext
from fhe_ml.ckks.containers.plaintext import PlaintextVector
from fhe_ml.ckks.containers.tensor import PlaintextTensor
from fhe_ml.utils.errors import ShapeError

if TYPE_CHECKING:
    from fhe_ml.ckks.context import FHEContext

class EncryptedVector:
    context: "FHEContext"
    _ct: CKKSCiphertext
    _n_values: int

    def __init__(
        self,
        context: "FHEContext",
        ct: CKKSCiphertext,
        n_values: int,
    ) -> None:
        self.context = context
        self._ct = ct
        self._n_values = n_values

    @property
    def size(self) -> int:
        return self._n_values

    @property
    def level(self) -> int:
        """Remaining usable multiplication levels in this ciphertext."""
        return self._ct.level

    def copy(self) -> "EncryptedVector":
        return EncryptedVector(self.context, self._ct.copy(), self._n_values)

    def mod_drop_to(self, target_level: int) -> "EncryptedVector":
        """Drop modulus primes until `self.level == target_level` (no-op if already)."""
        res = self.copy()
        while res._ct.level > target_level:
            self.context.mod_drop_inplace(res._ct)
        return res

    def rotate(self, k: int) -> "EncryptedVector":
        return self.context.rotate(self, k)

    def matmul(self, matrix: PlaintextTensor) -> "EncryptedVector":
        if not isinstance(matrix, PlaintextTensor):
            raise TypeError(f"Expected PlaintextTensor, got {type(matrix).__name__}")
        if matrix._encoded_diagonals is None:
            raise RuntimeError(
                "PlaintextTensor has not been encoded. Call "
                "Sequential.compile(context) before inference, or "
                "PlaintextTensor.encode(context) for standalone use."
            )
        n1, n2, (out_features, in_features) = matrix.meta
        if in_features != self._n_values:
            raise ShapeError(
                f"Matrix columns {in_features} != vector size {self._n_values}"
            )

        diagonals = matrix._encoded_diagonals
        target_depth = self._ct.depth

        baby: list[EncryptedVector] = [self.copy()]
        for _ in range(1, n1):
            baby.append(self.context.rotate(baby[-1], 1))

        result: EncryptedVector | None = None
        for j in range(n2 - 1, -1, -1):
            shift = n1 * j
            block: EncryptedVector | None = None

            for k in range(n1):
                i = shift + k
                stored_pt = diagonals[i] if i < len(diagonals) else None
                if stored_pt is None:
                    continue
                pt = stored_pt.copy()

                while pt.depth < target_depth:
                    self.context.mod_drop_plain_inplace(pt)

                term = baby[k].copy()
                self.context.multiply_plain_rescale(term._ct, pt)
                block = term if block is None else block + term

            if result is not None:
                result = self.context.rotate(result, n1)
            if block is not None:
                result = block if result is None else result + block

        if result is None:
            raise ShapeError("All matrix diagonals are zero")
        return EncryptedVector(self.context, result._ct, out_features)

    def __add__(
        self, other: Union["EncryptedVector", PlaintextVector, list[float], float]
    ) -> "EncryptedVector":
        res = self.copy()
        if isinstance(other, EncryptedVector):
            self.context.add_inplace(res._ct, other._ct.copy())
        else:
            self.context.add_plain_inplace(res._ct, self._resolve_plain(other))
        return res

    def __sub__(
        self, other: Union["EncryptedVector", PlaintextVector, list[float], float]
    ) -> "EncryptedVector":
        res = self.copy()
        if isinstance(other, EncryptedVector):
            self.context.sub_inplace(res._ct, other._ct.copy())
        else:
            self.context.sub_plain_inplace(res._ct, self._resolve_plain(other))
        return res

    def __mul__(
        self, other: Union["EncryptedVector", PlaintextVector, list[float], float]
    ) -> "EncryptedVector":
        res = self.copy()
        if isinstance(other, EncryptedVector):
            self.context.multiply_inplace(res._ct, other._ct.copy())
            self.context.relinearize_inplace(res._ct)
            self.context.rescale_inplace(res._ct)
        else:
            self.context.multiply_plain_rescale(res._ct, self._resolve_plain(other))
        return res

    def __radd__(
        self, other: Union["EncryptedVector", PlaintextVector, list[float], float]
    ) -> "EncryptedVector":
        return self.__add__(other)

    def __rsub__(
        self, other: Union["EncryptedVector", PlaintextVector, list[float], float]
    ) -> "EncryptedVector":
        return (self * -1).__add__(other)

    def __rmul__(
        self, other: Union["EncryptedVector", PlaintextVector, list[float], float]
    ) -> "EncryptedVector":
        return self.__mul__(other)

    def _resolve_plain(
        self, other: PlaintextVector | list[float] | float
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

    def _encode_and_align(self, values: list[float] | float) -> CKKSPlaintext:
        if isinstance(values, (int, float)):
            values_list: list[float] = [float(values)] * self._n_values
        else:
            values_list = list(values)
        pt = self.context.encode(values_list)
        while pt._pt.depth < self._ct.depth:
            self.context.mod_drop_plain_inplace(pt._pt)
        return pt._pt
