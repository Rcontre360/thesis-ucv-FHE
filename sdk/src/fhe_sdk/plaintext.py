from typing import TYPE_CHECKING, List

if TYPE_CHECKING:
    from fhe_sdk.context import FHEContext
    from fhe_sdk._backend import CKKSPlaintext

class Plaintext:
    def __init__(self, context: "FHEContext", backend_pt: "CKKSPlaintext", original_size: int):
        self._context = context
        self._backend_pt = backend_pt
        self._original_size = original_size

    @property
    def size(self) -> int:
        """Number of slots (equal to the length of the original values passed to ctx.encode())."""
        return self._original_size

    def decode(self) -> List[float]:
        """Shorthand for context.decode(self)."""
        return self._context.decode(self)

    def __repr__(self) -> str:
        return f"Plaintext(size={self._original_size}, depth={self._backend_pt.depth})"
