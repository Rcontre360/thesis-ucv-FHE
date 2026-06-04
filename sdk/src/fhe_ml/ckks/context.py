import math
from typing import List, Optional, Union

from fhe_ml.backend._backend import (
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
    BootstrappingConfig,
    BootstrappingType,
)
from fhe_ml.ckks.containers.plaintext import PlaintextVector
from fhe_ml.ckks.containers.ciphertext import EncryptedVector
from fhe_ml.ckks.config import FHEConfig


class FHEContext:
    config: FHEConfig
    _built: bool
    _bootstrapping_ready: bool

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

    def __init__(self, config: Optional[FHEConfig] = None) -> None:
        self.config = config if config is not None else FHEConfig()
        self._built = False
        self._bootstrapping_ready = False

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

    def build(self) -> "FHEContext":
        if self._built:
            raise RuntimeError("Context already built — create a new FHEContext to change parameters.")
        cfg = self.config

        self._backend_ctx = create_ckks_context_with_security(cfg.security_level)
        self._backend_ctx.set_poly_modulus_degree(1 << cfg.log_n)

        q_bits = cfg.coeff_modulus_bit_sizes[:-1]
        p_size = cfg.coeff_modulus_bit_sizes[-1]
        num_p = max(2, round(sum(q_bits) / (8 * p_size)))
        p_bits = [p_size] * num_p
        self._backend_ctx.set_coeff_modulus_bit_sizes(q_bits, p_bits)
        self._backend_ctx.generate()

        self._keygen = CKKSKeyGenerator(self._backend_ctx)
        self._sk = CKKSSecretkey(self._backend_ctx)
        self._keygen.generate_secret_key(self._sk)
        self._pk = CKKSPublickey(self._backend_ctx)
        self._keygen.generate_public_key(self._pk, self._sk)
        self._rk = CKKSRelinkey(self._backend_ctx)
        self._keygen.generate_relin_key(self._rk, self._sk)

        self._gk = CKKSGaloiskey(self._backend_ctx, self._network_shifts())
        self._generate_galois_key(self._gk)

        self._encoder = CKKSEncoder(self._backend_ctx)
        self._encryptor = CKKSEncryptor(self._backend_ctx, self._pk)
        self._decryptor = CKKSDecryptor(self._backend_ctx, self._sk)
        self._ops = CKKSOperator(self._backend_ctx, self._encoder)

        self._built = True
        return self

    @classmethod
    def default(cls) -> "FHEContext":
        return cls().build()

    def rotate(self, ct: EncryptedVector, k: int) -> EncryptedVector:
        if not self._built:
            raise RuntimeError("Context must be built before rotating.")
        result_ct = self._ops.rotate_rows(ct._ct, self._gk, k)
        return EncryptedVector(self, result_ct, ct._n_values)

    def encode(self, values: List[float]) -> PlaintextVector:
        if not self._built:
            raise RuntimeError("Context must be built before encoding.")
        n = len(values)
        if n == 0:
            raise ValueError("Cannot encode empty vector")
        slot_count = 1 << (self.config.log_n - 1)
        if n > slot_count:
            raise ValueError(f"Vector length {n} exceeds slot count {slot_count}")
        replicated = [values[k % n] for k in range(slot_count)]
        pt = CKKSPlaintext(self._backend_ctx)
        self._encoder.encode(pt, replicated, 2 ** self.config.log_scale)
        return PlaintextVector(self, pt, n)

    def decode(self, plaintext: PlaintextVector) -> List[float]:
        if not self._built:
            raise RuntimeError("Context must be built before decoding.")
        decoded = self._encoder.decode(plaintext._pt)
        return decoded[:plaintext.size]

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

    def _network_shifts(self) -> List[int]:
        slot_count = 1 << (self.config.log_n - 1)
        return [2**k for k in range(int(math.log2(slot_count)))]

    def _usable_levels(self) -> int:
        return self.encrypt([0.0])._ct.level

    def _setup_bootstrapping(self) -> None:
        if self._bootstrapping_ready:
            return
        if self.config.bootstrap is None:
            raise RuntimeError(
                "Bootstrapping required but FHEConfig.bootstrap is None. "
                "Set bootstrap=BootstrapConfig(...) on the FHEConfig."
            )
        boot = self.config.bootstrap
        config = BootstrappingConfig(
            boot.ctos_piece, boot.stoc_piece, boot.taylor_number, True
        )
        self._ops.generate_bootstrapping_params(
            2 ** self.config.log_scale, config, BootstrappingType.SLIM
        )
        boot_shifts = self._ops.bootstrapping_key_indexs()
        all_shifts = sorted(set(self._network_shifts()) | set(boot_shifts))
        gk = CKKSGaloiskey(self._backend_ctx, all_shifts)
        self._generate_galois_key(gk)
        self._gk = gk
        self._bootstrapping_ready = True

    def _generate_galois_key(self, gk) -> None:
        try:
            self._keygen.generate_galois_key(gk, self._sk, self.config.galois_keys_on_host)
        except TypeError:
            self._keygen.generate_galois_key(gk, self._sk)

    def _usable_after_boot(self) -> int:
        return self._bootstrap(self.encrypt([0.0]))._ct.level

    def _bootstrap(self, ct: EncryptedVector) -> EncryptedVector:
        if not self._bootstrapping_ready:
            raise RuntimeError("_setup_bootstrapping() must run before _bootstrap().")
        stoc = self.config.bootstrap.stoc_piece
        if ct.level < stoc:
            raise RuntimeError(
                f"Ciphertext at level {ct.level} is below the SLIM bootstrap "
                f"input requirement ({stoc}) — a refresh was needed earlier. "
                "Reduce activation depth (smaller ReLU `degrees`)."
            )
        raw = ct._ct.copy()
        while raw.level > stoc:
            self._ops.mod_drop_inplace(raw)
        refreshed = self._ops.slim_bootstrapping(raw, self._gk, self._rk)
        return EncryptedVector(self, refreshed, ct._n_values)

    def _prepare_for(self, ct: EncryptedVector, needed: int) -> EncryptedVector:
        if not self._bootstrapping_ready:
            return ct
        stoc = self.config.bootstrap.stoc_piece
        if ct.level >= needed + stoc:
            return ct
        refreshed = self._bootstrap(ct)
        if refreshed.level < needed:
            raise RuntimeError(
                f"A SLIM bootstrap restores {refreshed.level} levels but an "
                f"operation needs {needed} — reduce activation depth or "
                "lengthen coeff_modulus_bit_sizes."
            )
        return refreshed
