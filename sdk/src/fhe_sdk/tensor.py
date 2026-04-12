from typing import List, Tuple, Union


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

    @classmethod
    def from_numpy(cls, arr: object) -> "PlaintextTensor":
        """Construct from a numpy array (or any object with .tolist())."""
        if not hasattr(arr, "tolist"):
            raise TypeError(f"Expected an array-like with .tolist(), got {type(arr).__name__}")
        return cls(arr.tolist())  # type: ignore[union-attr]
