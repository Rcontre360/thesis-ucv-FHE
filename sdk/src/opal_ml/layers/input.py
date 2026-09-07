from opal_ml.ckks.containers.ciphertext import EncryptedVector
from opal_ml.client import Client


class Input:
    """Encrypted model input. Build via `model.input(client, data)`, not directly."""

    _ct: EncryptedVector
    _size: int

    def __init__(self, client: Client, flat_data: list[float]) -> None:
        self._ct = client.encrypt(flat_data)
        self._size = len(flat_data)

    @property
    def ciphertext(self) -> EncryptedVector:
        return self._ct

    @property
    def size(self) -> int:
        return self._size
