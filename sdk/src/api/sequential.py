from typing import TYPE_CHECKING, Callable, List

if TYPE_CHECKING:
    from api.ciphertext import EncryptedVector

Layer = Callable[["EncryptedVector"], "EncryptedVector"]


class Sequential:
    """Chain of layers and activation functions for encrypted vectors.

    Each element must be callable: layer(x) -> EncryptedVector.

    Example:
        model = Sequential([
            Linear(4, 8, weights, bias),
            ReLU(),
            Linear(8, 2, weights2),
        ])
        output = model(encrypted_input)
    """

    def __init__(self, layers: List[Layer]) -> None:
        if not layers:
            raise ValueError("Sequential requires at least one layer")
        self._layers = list(layers)

    def __call__(self, x: "EncryptedVector") -> "EncryptedVector":
        for layer in self._layers:
            x = layer(x)
        return x
