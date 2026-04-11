from typing import TYPE_CHECKING, List

if TYPE_CHECKING:
    from fhe_sdk.context import FHEContext
    from fhe_sdk._backend import CKKSCiphertext

class Ciphertext:
    def __init__(self, context: "FHEContext", backend_ct: "CKKSCiphertext", original_size: int):
        self._context = context
        self._backend_ct = backend_ct
        self._original_size = original_size

    @property
    def size(self) -> int:
        """Number of slots (equal to the length of the original values passed to ctx.encrypt())."""
        return self._original_size

    def decrypt(self) -> List[float]:
        """Shorthand for context.decrypt(self)."""
        return self._context.decrypt(self)

    def __repr__(self) -> str:
        return f"Ciphertext(size={self._original_size}, level={self._backend_ct.level})"
