import numpy as np

from fhe_ml.utils.errors import ShapeError


def check_array(
    data: object,
    *,
    ndim: int | None = None,
    shape: tuple[int, ...] | None = None,
    name: str = "input",
) -> np.ndarray:
    """Convert array-like `data` to a float ndarray, validating rank/shape.

    Raises `ShapeError` on non-numeric, ragged, or mismatched data.
    """
    try:
        arr = np.asarray(data, dtype=float)
    except (ValueError, TypeError) as e:
        raise ShapeError(f"{name} must be numeric and rectangular: {e}") from e
    if ndim is not None and arr.ndim != ndim:
        raise ShapeError(f"{name} must be {ndim}-D, got {arr.ndim}-D")
    if shape is not None and arr.shape != shape:
        raise ShapeError(f"{name} shape {arr.shape} != {shape}")
    return arr


def infer_shape(data: object) -> tuple[int, ...]:
    if not isinstance(data, list):
        return ()
    if len(data) == 0:
        return (0,)
    return (len(data),) + infer_shape(data[0])


def validate_shape(data: list, shape: tuple[int, ...]) -> None:
    if len(shape) == 1:
        for j, elem in enumerate(data):
            if not isinstance(elem, (int, float)):
                raise ValueError(
                    f"Expected numeric value at leaf index [{j}], "
                    f"got {type(elem).__name__}"
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
        validate_shape(row, shape[1:])
