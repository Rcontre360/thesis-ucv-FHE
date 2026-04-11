from typing import List, Union, Optional
from fhe_sdk._backend import (
    create_ckks_context,
    CKKSEncoder,
    CKKSEncryptor,
    CKKSDecryptor,
    CKKSKeyGenerator,
    CKKSSecretkey,
    CKKSPublickey,
    CKKSRelinkey,
    CKKSPlaintext,
    CKKSCiphertext,
    SecurityLevel as BackendSecurityLevel
)
from fhe_sdk.enums import SecurityLevel, KeyswitchingType
from fhe_sdk.plaintext import Plaintext
from fhe_sdk.ciphertext import Ciphertext

class FHEContext:
    def __init__(self) -> None:
        self._poly_modulus_degree: Optional[int] = None
        self._coeff_modulus_bit_sizes: Optional[List[int]] = None
        self._scale: Optional[float] = None
        self._security_level = SecurityLevel.SEC128
        self._keyswitching_type = KeyswitchingType.METHOD_I
        
        self._built = False
        
        # Backend objects
        self._backend_ctx = None
        self._encoder = None
        self._encryptor = None
        self._decryptor = None
        self._keygen = None
        
        # Keys
        self._sk = None
        self._pk = None
        self._rk = None

    def _ensure_not_built(self):
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

    def set_keyswitching_type(self, ktype: KeyswitchingType) -> "FHEContext":
        self._ensure_not_built()
        self._keyswitching_type = ktype
        return self

    def build(self) -> "FHEContext":
        if self._poly_modulus_degree is None:
            raise ValueError("poly_modulus_degree must be set")
        if self._coeff_modulus_bit_sizes is None:
            raise ValueError("coeff_modulus_bit_sizes must be set")
        if self._scale is None:
            raise ValueError("scale must be set")

        # Note: The current binary only seems to expose create_ckks_context()
        # which likely uses default SEC128. 
        self._backend_ctx = create_ckks_context()
        self._backend_ctx.set_poly_modulus_degree(self._poly_modulus_degree)
        
        # Split bit_sizes into Q and P
        # Logic: last prime is P
        q_bits = self._coeff_modulus_bit_sizes[:-1]
        p_bits = [self._coeff_modulus_bit_sizes[-1]]

        self._backend_ctx.set_coeff_modulus_bit_sizes(q_bits, p_bits)
        self._backend_ctx.generate()

        # Initialize keys and generators
        self._keygen = CKKSKeyGenerator(self._backend_ctx)
        self._sk = CKKSSecretkey(self._backend_ctx)
        self._keygen.generate_secret_key(self._sk)
        
        self._pk = CKKSPublickey(self._backend_ctx)
        self._keygen.generate_public_key(self._pk, self._sk)
        
        self._rk = CKKSRelinkey(self._backend_ctx)
        self._keygen.generate_relin_key(self._rk, self._sk)
        
        # Initialize encoder/crypto
        self._encoder = CKKSEncoder(self._backend_ctx)
        self._encryptor = CKKSEncryptor(self._backend_ctx, self._pk)
        self._decryptor = CKKSDecryptor(self._backend_ctx, self._sk)
        
        self._built = True
        return self

    @classmethod
    def default(cls) -> "FHEContext":
        return (
            cls()
            .set_poly_modulus_degree(8192)
            .set_coeff_modulus_bit_sizes([60, 40, 40, 60])
            .set_scale(2**40)
            .set_security_level(SecurityLevel.SEC128)
            .set_keyswitching_type(KeyswitchingType.METHOD_I)
            .build()
        )

    def encode(self, values: List[float]) -> Plaintext:
        if not self._built:
            raise RuntimeError("Context must be built before encoding.")
        
        backend_pt = CKKSPlaintext(self._backend_ctx)
        self._encoder.encode(backend_pt, values, self._scale)
        return Plaintext(self, backend_pt, len(values))

    def decode(self, plaintext: Plaintext) -> List[float]:
        if not self._built:
            raise RuntimeError("Context must be built before decoding.")
        if plaintext._context is not self:
            raise ValueError("Plaintext was produced by a different FHEContext.")
            
        decoded = self._encoder.decode(plaintext._backend_pt)
        return decoded[:plaintext.size]

    def encrypt(self, values: Union[List[float], Plaintext]) -> Ciphertext:
        if not self._built:
            raise RuntimeError("Context must be built before encrypting.")
            
        if isinstance(values, list):
            plaintext = self.encode(values)
        else:
            plaintext = values
            
        if plaintext._context is not self:
            raise ValueError("Plaintext was produced by a different FHEContext.")
            
        backend_ct = CKKSCiphertext(self._backend_ctx)
        self._encryptor.encrypt(backend_ct, plaintext._backend_pt)
        return Ciphertext(self, backend_ct, plaintext.size)

    def decrypt(self, ciphertext: Ciphertext) -> List[float]:
        if not self._built:
            raise RuntimeError("Context must be built before decrypting.")
        if ciphertext._context is not self:
            raise ValueError("Ciphertext was produced by a different FHEContext.")
            
        backend_pt = CKKSPlaintext(self._backend_ctx)
        self._decryptor.decrypt(backend_pt, ciphertext._backend_ct)
        
        decoded = self._encoder.decode(backend_pt)
        return decoded[:ciphertext.size]
