import numpy as np


def to_numpy(batch: object) -> np.ndarray:
    """Coerce a data batch to a float numpy array.

    Accepts a plain array-like, a torch tensor (detached automatically), or a
    `(inputs, targets)` pair as a PyTorch DataLoader yields — in which case the
    inputs (first element) are taken.
    """
    if isinstance(batch, (tuple, list)):
        batch = batch[0]
    if hasattr(batch, "detach"):  # torch tensor
        batch = batch.detach().cpu().numpy()
    return np.asarray(batch, dtype=float)
