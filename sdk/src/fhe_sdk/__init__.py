"""fhe_sdk — Python bindings for HEonGPU CKKS scheme.

Re-exports every public symbol from the compiled _backend extension module.

Symbols intentionally NOT exported:
  - SecurityLevel.NONE  (not bound in _backend, per ERROR-005)
  - KeyswitchingType    (not exposed; keyswitching is inferred from P vector
                         size passed to set_coeff_modulus_bit_sizes, per ERROR-006)
  - Scheme              (BFV/TFHE are not bound; only CKKS bindings are present)
"""

from fhe_sdk._backend import (
    # Security level enum (SEC128, SEC192, SEC256 only — NONE is not exposed)
    SecurityLevel,
    # Context
    CKKSContext,
    create_ckks_context,
    create_ckks_context_with_security,
    # Keys
    CKKSSecretkey,
    CKKSPublickey,
    CKKSRelinkey,
    CKKSGaloiskey,
    # Key generation
    CKKSKeyGenerator,
    # Data types
    CKKSPlaintext,
    CKKSCiphertext,
    # Cryptographic primitives
    CKKSEncoder,
    CKKSEncryptor,
    CKKSDecryptor,
    CKKSOperator,
)

__all__ = [
    "SecurityLevel",
    "CKKSContext",
    "create_ckks_context",
    "create_ckks_context_with_security",
    "CKKSSecretkey",
    "CKKSPublickey",
    "CKKSRelinkey",
    "CKKSGaloiskey",
    "CKKSKeyGenerator",
    "CKKSPlaintext",
    "CKKSCiphertext",
    "CKKSEncoder",
    "CKKSEncryptor",
    "CKKSDecryptor",
    "CKKSOperator",
]
