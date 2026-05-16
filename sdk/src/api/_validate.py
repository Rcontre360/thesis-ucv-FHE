from typing import Optional, Tuple

import numpy as np

from api.errors import ShapeError


def check_array(
    data: object,
    *,
    ndim: Optional[int] = None,
    shape: Optional[Tuple[int, ...]] = None,
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
