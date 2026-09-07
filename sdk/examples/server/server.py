"""Server side: holds the evaluator context and runs the encrypted computation.

The server never sees the secret key. The client hands it the public evaluation
keys (relin + galois) via `key_params`; from then on it can multiply, rotate and
add ciphertexts without ever decrypting them.
"""

from collections.abc import Callable

from opal_ml import EncryptedVector, FHEContext, KeyParams

WEIGHTS: list[float] = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
SCALE: float = 2.0
REDUCTION_SHIFTS: tuple[int, ...] = (1, 2, 4)


def start_server(
    context: FHEContext, key_params: KeyParams
) -> Callable[[EncryptedVector], EncryptedVector]:
    context.set_client_params(key_params)

    def handle(request: EncryptedVector) -> EncryptedVector:
        result = request * WEIGHTS  # encrypted * plaintext weights
        result = result * SCALE  # encrypted * scalar
        for shift in REDUCTION_SHIFTS:  # rotate-and-add reduction over the block
            result = result + result.rotate(shift)
        return result

    return handle
