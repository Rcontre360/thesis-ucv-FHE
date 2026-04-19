from typing import List, Optional, Tuple, Union


def _infer_shape(data: object) -> Tuple[int, ...]:
    if not isinstance(data, list):
        return ()
    if len(data) == 0:
        return (0,)
    return (len(data),) + _infer_shape(data[0])


def _validate_shape(data: list, shape: Tuple[int, ...]) -> None:
    if len(shape) == 1:
        for j, elem in enumerate(data):
            if not isinstance(elem, (int, float)):
                raise ValueError(
                    f"Expected numeric value at leaf index [{j}], got {type(elem).__name__}"
                )
        return
    for i, row in enumerate(data):
        if not isinstance(row, list):
            raise ValueError(f"Expected list at index [{i}], got {type(row).__name__}")
        if len(row) != shape[1]:
            raise ValueError(
                f"Dimension mismatch at index [{i}]: "
                f"expected {shape[1]} elements, got {len(row)}"
            )
        _validate_shape(row, shape[1:])


class PlaintextTensor:
    """
    Plaintext tensor for 2D or 3D data backed by a nested Python list.

    2D  (rows, cols)         — e.g. weight matrix for a Linear layer
    3D  (depth, rows, cols)  — e.g. conv filter bank or batched matrix
    """

    _data: List
    _shape: Tuple[int, ...]

    def __init__(self, data: List) -> None:
        shape = _infer_shape(data)
        if len(shape) not in (2, 3):
            raise ValueError(
                f"PlaintextTensor requires 2D or 3D data, got {len(shape)}D. "
                "Pass a nested list of depth 2 (matrix) or 3 (cube)."
            )
        if any(s == 0 for s in shape):
            raise ValueError(f"All dimensions must be non-zero, got shape {shape}.")
        _validate_shape(data, shape)
        self._data = data
        self._shape = shape

    @property
    def shape(self) -> Tuple[int, ...]:
        return self._shape

    @property
    def ndim(self) -> int:
        return len(self._shape)

    def __len__(self) -> int:
        return self._shape[0]

    def __getitem__(self, idx: int) -> Union[List[float], "PlaintextTensor"]:
        """Return row idx.  For 2D returns List[float]; for 3D returns a PlaintextTensor."""
        if idx < 0 or idx >= self._shape[0]:
            raise IndexError(
                f"Index {idx} out of range for dimension 0 of size {self._shape[0]}"
            )
        row = self._data[idx]
        if self.ndim == 2:
            return row
        return PlaintextTensor(row)

    def __repr__(self) -> str:
        return f"PlaintextTensor(shape={self._shape})"

    def get_diagonal(
        self,
        k: int,
        max_size: Optional[int] = None,
        axis: int = 0,
        slice_index: int = 0,
    ) -> List[float]:
        """
        Return the k-th rotated diagonal of a 2D slice of this tensor.

        For a 2D tensor the slice is the tensor itself; axis and slice_index
        are ignored.

        For a 3D tensor of shape (D, R, C):
          axis=0, slice_index=i  →  matrix data[i, :, :]   shape (R, C)
          axis=1, slice_index=j  →  matrix data[:, j, :]   shape (D, C)
          axis=2, slice_index=k_ →  matrix data[:, :, k_]  shape (D, R)

        k=0 is the main diagonal; k>0 shifts toward upper diagonals; k<0
        toward lower diagonals.  The diagonal wraps cyclically, which is the
        convention required by the diagonal matmul algorithm.

        max_size caps the number of elements returned (default: n_rows * n_cols).
        """
        if self.ndim == 2:
            n_rows, n_cols = self._shape
            mat: List[List[float]] = self._data  # type: ignore[assignment]
        else:
            depth, rows, cols = self._shape
            if axis == 0:
                if not (0 <= slice_index < depth):
                    raise IndexError(
                        f"slice_index {slice_index} out of range for axis 0 (size {depth})"
                    )
                mat = self._data[slice_index]  # type: ignore[assignment]
                n_rows, n_cols = rows, cols
            elif axis == 1:
                if not (0 <= slice_index < rows):
                    raise IndexError(
                        f"slice_index {slice_index} out of range for axis 1 (size {rows})"
                    )
                mat = [
                    [self._data[d][slice_index][c] for c in range(cols)]
                    for d in range(depth)
                ]
                n_rows, n_cols = depth, cols
            elif axis == 2:
                if not (0 <= slice_index < cols):
                    raise IndexError(
                        f"slice_index {slice_index} out of range for axis 2 (size {cols})"
                    )
                mat = [
                    [self._data[d][r][slice_index] for r in range(rows)]
                    for d in range(depth)
                ]
                n_rows, n_cols = depth, rows
            else:
                raise ValueError(f"axis must be 0, 1, or 2 for a 3D tensor, got {axis}")

        return self._diagonal_of(mat, n_rows, n_cols, k, max_size)

    @staticmethod
    def _diagonal_of(
        mat: List[List[float]],
        n_rows: int,
        n_cols: int,
        k: int,
        max_size: Optional[int],
    ) -> List[float]:
        r_offset = 0 if k >= 0 else -k
        c_offset = k if k >= 0 else 0
        natural_size = min(n_rows, n_cols)
        size = min(max_size, n_rows * n_cols) if max_size is not None else natural_size
        return [
            mat[(r_offset + i) % n_rows][(c_offset + i) % n_cols]
            for i in range(size)
        ]

    @classmethod
    def from_numpy(cls, arr: object) -> "PlaintextTensor":
        """Construct from a numpy array (or any object with .tolist())."""
        if not hasattr(arr, "tolist"):
            raise TypeError(f"Expected an array-like with .tolist(), got {type(arr).__name__}")
        return cls(arr.tolist())  # type: ignore[union-attr]
