from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, List

if TYPE_CHECKING:
    from api.ciphertext import EncryptedVector


class Layer(ABC):
    """Interface for `Sequential` elements: a callable EncryptedVector -> EncryptedVector.

    Subclasses must implement `__call__`. Layers usable as a model's first
    element override `prepare_input`; others inherit the raising default.
    """

    @abstractmethod
    def __call__(self, x: "EncryptedVector") -> "EncryptedVector":
        ...

    def prepare_input(self, raw_data: object) -> List[float]:
        raise NotImplementedError(
            f"{type(self).__name__} cannot be a model's input layer; "
            "use a weighted layer (Linear, Conv2D, ...) as the first element."
        )
