import math
from typing import List, Optional, Union

from core._backend import (
    create_ckks_context_with_security,
    CKKSEncoder,
    CKKSEncryptor,
    CKKSDecryptor,
    CKKSKeyGenerator,
    CKKSSecretkey,
    CKKSPublickey,
    CKKSRelinkey,
    CKKSGaloiskey,
    CKKSPlaintext,
    CKKSCiphertext,
    CKKSOperator,
)
from core.enums import SecurityLevel
from core.plaintext import PlaintextVector
from api.ciphertext import EncryptedVector


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
    _gk: Optional[CKKSGaloiskey]

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
        self._gk = None

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

        # Last element is the P prime (key-switching auxiliary modulus).
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

        # Galois key: powers-of-2 shifts from 1 to slot_count.
        # Covers both _sum_slots (shifts 1..n/2) and _replicate_slot0 (shifts slot_count/2..1).
        slot_count = self._poly_modulus_degree // 2
        shifts = [2**k for k in range(int(math.log2(slot_count)))]
        self._gk = CKKSGaloiskey(self._backend_ctx, shifts)
        self._keygen.generate_galois_key(self._gk, self._sk)

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
            .set_poly_modulus_degree(16384)
            .set_coeff_modulus_bit_sizes([60, 40, 40, 40, 40, 60])
            .set_scale(2**40)
            .build()
        )

    # ------------------------------------------------------------------
    # Rotation
    # ------------------------------------------------------------------

    def rotate(self, ct: EncryptedVector, k: int) -> EncryptedVector:
        if not self._built:
            raise RuntimeError("Context must be built before rotating.")
        result_ct = self._ops.rotate_rows(ct._ct, self._gk, k)
        return EncryptedVector(self, result_ct, ct._n_values)

    # ------------------------------------------------------------------
    # Encode / decode
    # ------------------------------------------------------------------

    def encode(self, values: List[float]) -> PlaintextVector:
        if not self._built:
            raise RuntimeError("Context must be built before encoding.")
        n = len(values)
        if n == 0:
            raise ValueError("Cannot encode empty vector")
        slot_count = self._poly_modulus_degree // 2
        if n > slot_count:
            raise ValueError(f"Vector length {n} exceeds slot count {slot_count}")
        # Replicate values cyclically across all slots: replicated[k] = values[k mod n].
        # The cyclic-wrap diagonal matmul (à la TenSEAL) relies on every slot
        # carrying x[k mod n], not zeros past index n.
        replicated = [values[k % n] for k in range(slot_count)]
        pt = CKKSPlaintext(self._backend_ctx)
        self._encoder.encode(pt, replicated, self._scale)
        return PlaintextVector(self, pt, n)

    def decode(self, plaintext: PlaintextVector) -> List[float]:
        if not self._built:
            raise RuntimeError("Context must be built before decoding.")
        decoded = self._encoder.decode(plaintext._pt)
        return decoded[:plaintext.size]

    # ------------------------------------------------------------------
    # Encrypt / decrypt
    # ------------------------------------------------------------------

    def encrypt(self, values: Union[List[float], PlaintextVector]) -> EncryptedVector:
        if not self._built:
            raise RuntimeError("Context must be built before encrypting.")
        if isinstance(values, list):
            plaintext = self.encode(values)
        else:
            plaintext = values
        ct = CKKSCiphertext(self._backend_ctx)
        self._encryptor.encrypt(ct, plaintext._pt)
        return EncryptedVector(self, ct, plaintext.size)

    def decrypt(self, ciphertext: EncryptedVector) -> List[float]:
        if not self._built:
            raise RuntimeError("Context must be built before decrypting.")
        pt = CKKSPlaintext(self._backend_ctx)
        self._decryptor.decrypt(pt, ciphertext._ct)
        decoded = self._encoder.decode(pt)
        return decoded[:ciphertext.size]
