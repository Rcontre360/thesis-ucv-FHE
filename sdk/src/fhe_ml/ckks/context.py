from fhe_ml.backend import *
from fhe_ml.ckks.config import FHEConfig
from fhe_ml.ckks.containers.ciphertext import EncryptedVector
from fhe_ml.ckks.containers.plaintext import PlaintextVector
from fhe_ml.ckks.keyparams import KeyParams


class FHEContext:
    """Server-side / evaluator context: parameters + model ops, no secret key.

    Holds only public material. Client keys (relin + galois) are injected via
    `set_client_params`; key generation, encryption and decryption live in
    `Client`.
    """

    config: FHEConfig
    _bootstrapping_ready: bool

    _backend_ctx: object
    _encoder: CKKSEncoder
    _ops: CKKSOperator

    _rk: CKKSRelinkey | None
    _gk: CKKSGaloiskey | None

    def __init__(self, config: FHEConfig) -> None:
        self.config = config
        self._bootstrapping_ready = False
        self._rk = None
        self._gk = None

        self._backend_ctx = create_ckks_context_with_security(config.security_level)
        self._backend_ctx.set_poly_modulus_degree(1 << config.log_n)

        q_bits = config.coeff_modulus_bit_sizes[:-1]
        p_size = config.coeff_modulus_bit_sizes[-1]
        num_p = max(2, round(sum(q_bits) / (8 * p_size)))
        p_bits = [p_size] * num_p

        self._backend_ctx.set_coeff_modulus_bit_sizes(q_bits, p_bits)
        self._backend_ctx.generate()

        self._encoder = CKKSEncoder(self._backend_ctx)
        self._ops = CKKSOperator(self._backend_ctx, self._encoder)

    @classmethod
    def default(cls) -> "FHEContext":
        return cls(FHEConfig())

    def set_client_params(self, params: KeyParams) -> None:
        self._rk = params.relin_key
        if params.galois_key is not None:
            self._gk = params.galois_key

    def rotate(self, ct: EncryptedVector, k: int) -> EncryptedVector:
        result_ct = self._ops.rotate_rows(ct._ct, self._gk, k)
        return EncryptedVector(self, result_ct, ct._n_values)

    def multiply_plain_inplace(self, ct: CKKSCiphertext, pt: CKKSPlaintext) -> None:
        self._ops.multiply_plain_inplace(ct, pt)

    def rescale_inplace(self, ct: CKKSCiphertext) -> None:
        self._ops.rescale_inplace(ct)

    def multiply_plain_rescale(self, ct: CKKSCiphertext, pt: CKKSPlaintext) -> None:
        self.multiply_plain_inplace(ct, pt)
        self.rescale_inplace(ct)

    def multiply_inplace(self, ct: CKKSCiphertext, other: CKKSCiphertext) -> None:
        self._ops.multiply_inplace(ct, other)

    def relinearize_inplace(self, ct: CKKSCiphertext) -> None:
        self._ops.relinearize_inplace(ct, self._rk)

    def add_inplace(self, ct: CKKSCiphertext, other: CKKSCiphertext) -> None:
        self._ops.add_inplace(ct, other)

    def add_plain_inplace(self, ct: CKKSCiphertext, pt: CKKSPlaintext) -> None:
        self._ops.add_plain_inplace(ct, pt)

    def sub_inplace(self, ct: CKKSCiphertext, other: CKKSCiphertext) -> None:
        self._ops.sub_inplace(ct, other)

    def sub_plain_inplace(self, ct: CKKSCiphertext, pt: CKKSPlaintext) -> None:
        self._ops.sub_plain_inplace(ct, pt)

    def mod_drop_inplace(self, ct: CKKSCiphertext) -> None:
        self._ops.mod_drop_inplace(ct)

    def mod_drop_plain_inplace(self, pt: CKKSPlaintext) -> None:
        self._ops.mod_drop_plain_inplace(pt)

    def encode(self, values: list[float]) -> PlaintextVector:
        n = len(values)
        if n == 0:
            raise ValueError("Cannot encode empty vector")
        slot_count = 1 << (self.config.log_n - 1)
        if n > slot_count:
            raise ValueError(f"Vector length {n} exceeds slot count {slot_count}")
        replicated = [values[k % n] for k in range(slot_count)]
        pt = CKKSPlaintext(self._backend_ctx)
        self._encoder.encode(pt, replicated, 2**self.config.log_scale)
        return PlaintextVector(self, pt, n)

    def decode(self, plaintext: PlaintextVector) -> list[float]:
        decoded = self._encoder.decode(plaintext._pt)
        return decoded[: plaintext.size]

    def _usable_levels(self) -> int:
        # Deterministic from the chain (Q_size - 1); no ciphertext/key needed.
        return len(self.config.coeff_modulus_bit_sizes) - 2

    def _setup_bootstrapping(self) -> list[int]:
        if self.config.bootstrap is None:
            raise RuntimeError(
                "Bootstrapping required but FHEConfig.bootstrap is None. "
                "Set bootstrap=BootstrapConfig(...) on the FHEConfig."
            )
        if not self._bootstrapping_ready:
            boot = self.config.bootstrap
            config = BootstrappingConfig(
                boot.ctos_piece, boot.stoc_piece, boot.taylor_number, True
            )
            self._ops.generate_bootstrapping_params(
                2**self.config.log_scale, config, BootstrappingType.SLIM
            )
            self._bootstrapping_ready = True
        return list(self._ops.bootstrapping_key_indexs())

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
