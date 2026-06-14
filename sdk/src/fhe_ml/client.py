from collections.abc import Iterable

from fhe_ml.backend import *
from fhe_ml.ckks.containers.ciphertext import EncryptedVector
from fhe_ml.ckks.context import FHEContext
from fhe_ml.ckks.keyparams import KeyParams


class Client:
    def __init__(self, context: FHEContext) -> None:
        self._context = context
        backend = context._backend_ctx

        self._keygen = CKKSKeyGenerator(backend)
        self._sk = CKKSSecretkey(backend)
        self._keygen.generate_secret_key(self._sk)
        self._pk = CKKSPublickey(backend)
        self._keygen.generate_public_key(self._pk, self._sk)
        self._rk = CKKSRelinkey(backend)
        self._keygen.generate_relin_key(self._rk, self._sk)

        self._encryptor = CKKSEncryptor(backend, self._pk)
        self._decryptor = CKKSDecryptor(backend, self._sk)
        self._gk = None

        if context.config.galois_shifts:
            self.generate_rotation_keys(context.config.galois_shifts)

    def generate_rotation_keys(self, shifts: Iterable[int]) -> None:
        all_shifts = sorted({int(s) for s in shifts})
        if not all_shifts:
            raise ValueError("generate_rotation_keys requires at least one shift")
        gk = CKKSGaloiskey(self._context._backend_ctx, all_shifts)
        try:
            self._keygen.generate_galois_key(
                gk, self._sk, self._context.config.galois_keys_on_host
            )
        except TypeError:
            self._keygen.generate_galois_key(gk, self._sk)
        self._gk = gk

    def key_params(self) -> KeyParams:
        return KeyParams(relin_key=self._rk, galois_key=self._gk)

    def encrypt(self, values: list[float]) -> EncryptedVector:
        plaintext = self._context.encode(values)
        ct = CKKSCiphertext(self._context._backend_ctx)
        self._encryptor.encrypt(ct, plaintext._pt)
        return EncryptedVector(self._context, ct, plaintext.size)

    def decrypt(self, vector: EncryptedVector) -> list[float]:
        pt = CKKSPlaintext(self._context._backend_ctx)
        self._decryptor.decrypt(pt, vector._ct)
        decoded = self._context._encoder.decode(pt)
        return decoded[: vector.size]
