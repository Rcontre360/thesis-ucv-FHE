from typing import TYPE_CHECKING, List

from api.ciphertext import EncryptedVector

if TYPE_CHECKING:
    from api.context import FHEContext


class Input:
    """Encrypted model input. Build via `model.input(context, data)`, not directly."""

    _ct: EncryptedVector
    _size: int

    def __init__(self, context: "FHEContext", flat_data: List[float]) -> None:
        self._ct = context.encrypt(flat_data)
        self._size = len(flat_data)

    @property
    def ciphertext(self) -> EncryptedVector:
        return self._ct

    @property
    def size(self) -> int:
        return self._size
