from typing import TYPE_CHECKING, NamedTuple

import numpy as np

from fhe_ml.backend._backend import CKKSPlaintext
from fhe_ml.utils.validate import infer_shape, validate_shape

if TYPE_CHECKING:
    from fhe_ml.ckks.context import FHEContext


class TensorMeta(NamedTuple):
    n1: int
    n2: int
    shape: tuple[int, int]


class PlaintextTensor:
    """Plaintext weight matrix (2D) backed by a nested Python list."""

    _data: list
    _meta: TensorMeta
    _encoded_diagonals: list[CKKSPlaintext | None] | None

    def __init__(self, data: list) -> None:
        shape = infer_shape(data)
        if len(shape) != 2:
            raise ValueError(
                f"PlaintextTensor requires 2D data (a matrix), got {len(shape)}D. "
                "Pass a nested list of depth 2."
            )
        if any(s == 0 for s in shape):
            raise ValueError(f"All dimensions must be non-zero, got shape {shape}.")
        validate_shape(data, shape)
        n1, n2 = self._bsgs_factorization(shape[1])
        self._data = data
        self._meta = TensorMeta(n1=n1, n2=n2, shape=shape)
        self._encoded_diagonals = None

    @property
    def meta(self) -> TensorMeta:
        return self._meta

    @staticmethod
    def _bsgs_factorization(in_features: int) -> tuple[int, int]:
        n1 = (
            1 if in_features <= 1 else 1 << max(0, round(np.log2(np.sqrt(in_features))))
        )
        n2 = (in_features + n1 - 1) // n1
        return n1, n2

    def encode(self, context: "FHEContext") -> None:
        if self._encoded_diagonals is not None:
            return
        out_features, in_features = self._meta.shape
        n1, n2 = self._meta.n1, self._meta.n2
        slot_count = 1 << (context.config.log_n - 1)
        md = self.to_numpy()  # (out_features, in_features)
        s = np.arange(slot_count)
        s_out = s % out_features

        encoded: list[CKKSPlaintext | None] = [None] * (n1 * n2)
        for j in range(n2):
            shift = n1 * j
            for k in range(n1):
                i = shift + k
                if i >= in_features:
                    continue
                # full-slot diagonal, then fold in the giant rotation (-n1*j)
                diag = md[s_out, (i + s) % in_features]
                if not diag.any():
                    continue
                rotated = np.roll(diag, shift)
                encoded[i] = context.encode(rotated.tolist())._pt

        self._encoded_diagonals = encoded

    def bsgs_shifts(self) -> list[int]:
        n1, n2 = self._meta.n1, self._meta.n2
        shifts = [n1 * j for j in range(1, n2)]
        if n1 > 1:
            shifts.append(1)
        return shifts

    @property
    def shape(self) -> tuple[int, int]:
        return self._meta.shape

    @property
    def ndim(self) -> int:
        return len(self._meta.shape)

    def __len__(self) -> int:
        return self._meta.shape[0]

    def __repr__(self) -> str:
        return f"PlaintextTensor(shape={self._meta.shape})"

    @classmethod
    def from_numpy(cls, arr: object) -> "PlaintextTensor":
        """Construct from a numpy array (or any object with .tolist())."""
        if not hasattr(arr, "tolist"):
            raise TypeError(
                f"Expected an array-like with .tolist(), got {type(arr).__name__}"
            )
        return cls(arr.tolist())  # type: ignore[union-attr]

    def to_numpy(self) -> np.ndarray:
        """Return the tensor data as a float numpy array."""
        return np.asarray(self._data, dtype=float)
