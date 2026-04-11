from typing import List, Optional, Union

from fhe_sdk._backend import (
    create_ckks_context_with_security,
    CKKSEncoder,
    CKKSEncryptor,
    CKKSDecryptor,
    CKKSKeyGenerator,
    CKKSSecretkey,
    CKKSPublickey,
    CKKSRelinkey,
    CKKSPlaintext,
    CKKSCiphertext,
    CKKSOperator,
)
from fhe_sdk.enums import SecurityLevel
from fhe_sdk.plaintext import Plaintext
from fhe_sdk.ciphertext import Ciphertext


class FHEContext:
    _poly_modulus_degree: Optional[int]
    _coeff_modulus_bit_sizes: Optional[List[int]]
    _scale: Optional[float]
    _security_level: SecurityLevel
    _built: bool

    _backend_ctx: Optional[object]
    _encoder: Optional[CKKSEncoder]
    _encryptor: Optional[CKKSEncryptor]
    _decryptor: Optional[CKKSDecryptor]
    _keygen: Optional[CKKSKeyGenerator]
    _ops: Optional[CKKSOperator]

    _sk: Optional[CKKSSecretkey]
    _pk: Optional[CKKSPublickey]
    _rk: Optional[CKKSRelinkey]

    def __init__(self) -> None:
        self._poly_modulus_degree = None
        self._coeff_modulus_bit_sizes = None
        self._scale = None
        self._security_level = SecurityLevel.SEC128
        self._built = False

        self._backend_ctx = None
        self._encoder = None
        self._encryptor = None
        self._decryptor = None
        self._keygen = None
        self._ops = None

        self._sk = None
        self._pk = None
        self._rk = None

    def _ensure_not_built(self) -> None:
        if self._built:
            raise RuntimeError("Context already built — create a new FHEContext to change parameters.")

    def set_poly_modulus_degree(self, degree: int) -> "FHEContext":
        self._ensure_not_built()
        self._poly_modulus_degree = degree
        return self

    def set_coeff_modulus_bit_sizes(self, bit_sizes: List[int]) -> "FHEContext":
        self._ensure_not_built()
        self._coeff_modulus_bit_sizes = bit_sizes
        return self

    def set_scale(self, scale: float) -> "FHEContext":
        self._ensure_not_built()
        self._scale = scale
        return self

    def set_security_level(self, level: SecurityLevel) -> "FHEContext":
        self._ensure_not_built()
        self._security_level = level
        return self

    def build(self) -> "FHEContext":
        if self._poly_modulus_degree is None:
            raise ValueError("poly_modulus_degree must be set before build()")
        if self._coeff_modulus_bit_sizes is None:
            raise ValueError("coeff_modulus_bit_sizes must be set before build()")
        if self._scale is None:
            raise ValueError("scale must be set before build()")

        self._backend_ctx = create_ckks_context_with_security(self._security_level)
        self._backend_ctx.set_poly_modulus_degree(self._poly_modulus_degree)

        # Last prime is the P prime (key-switching auxiliary modulus).
        # One P prime => KEYSWITCHING_METHOD_I (inferred by HEonGPU from P vector size).
        q_bits = self._coeff_modulus_bit_sizes[:-1]
        p_bits = [self._coeff_modulus_bit_sizes[-1]]
        self._backend_ctx.set_coeff_modulus_bit_sizes(q_bits, p_bits)
        self._backend_ctx.generate()

        self._keygen = CKKSKeyGenerator(self._backend_ctx)
        self._sk = CKKSSecretkey(self._backend_ctx)
        self._keygen.generate_secret_key(self._sk)
        self._pk = CKKSPublickey(self._backend_ctx)
        self._keygen.generate_public_key(self._pk, self._sk)
        self._rk = CKKSRelinkey(self._backend_ctx)
        self._keygen.generate_relin_key(self._rk, self._sk)

        self._encoder = CKKSEncoder(self._backend_ctx)
        self._encryptor = CKKSEncryptor(self._backend_ctx, self._pk)
        self._decryptor = CKKSDecryptor(self._backend_ctx, self._sk)
        self._ops = CKKSOperator(self._backend_ctx, self._encoder)

        self._built = True
        return self

    @classmethod
    def default(cls) -> "FHEContext":
        return (
            cls()
            .set_poly_modulus_degree(8192)
            .set_coeff_modulus_bit_sizes([60, 40, 40, 60])
            .set_scale(2**40)
            .build()
        )

    def encode(self, values: List[float]) -> Plaintext:
        if not self._built:
            raise RuntimeError("Context must be built before encoding.")
        pt = CKKSPlaintext(self._backend_ctx)
        self._encoder.encode(pt, values, self._scale)
        return Plaintext(self, pt, len(values))

    def decode(self, plaintext: Plaintext) -> List[float]:
        if not self._built:
            raise RuntimeError("Context must be built before decoding.")
        decoded = self._encoder.decode(plaintext._pt)
        return decoded[:plaintext.size]

    def encrypt(self, values: Union[List[float], Plaintext]) -> Ciphertext:
        if not self._built:
            raise RuntimeError("Context must be built before encrypting.")
        if isinstance(values, list):
            plaintext = self.encode(values)
        else:
            plaintext = values
        ct = CKKSCiphertext(self._backend_ctx)
        self._encryptor.encrypt(ct, plaintext._pt)
        return Ciphertext(self, ct, plaintext.size)

    def decrypt(self, ciphertext: Ciphertext) -> List[float]:
        if not self._built:
            raise RuntimeError("Context must be built before decrypting.")
        pt = CKKSPlaintext(self._backend_ctx)
        self._decryptor.decrypt(pt, ciphertext._ct)
        decoded = self._encoder.decode(pt)
        return decoded[:ciphertext.size]
