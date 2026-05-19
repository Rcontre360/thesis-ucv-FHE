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
    BootstrappingConfig,
    BootstrappingType,
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

    # SLIM bootstrapping is always the variant used; these tune the circuit.
    _ctos_piece: int
    _stoc_piece: int
    _taylor_number: int
    _bootstrapping_ready: bool
    _galois_keys_on_host: bool

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

        self._ctos_piece = 3
        self._stoc_piece = 3
        self._taylor_number = 11
        self._bootstrapping_ready = False
        self._galois_keys_on_host = False

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

    def set_bootstrapping_params(
        self, ctos_piece: int = 3, stoc_piece: int = 3, taylor_number: int = 11
    ) -> "FHEContext":
        """Tune the SLIM bootstrapping circuit (used only if `compile` enables it).

        ctos_piece/stoc_piece (2-5): DFT factor counts — more pieces means a
        deeper but cheaper-per-stage circuit. taylor_number (6-15): EvalMod sine
        degree — higher is more accurate but deeper. Defaults are a middle ground.
        """
        self._ensure_not_built()
        self._ctos_piece = ctos_piece
        self._stoc_piece = stoc_piece
        self._taylor_number = taylor_number
        return self

    def set_galois_key_storage(self, on_host: bool = True) -> "FHEContext":
        """Keep Galois (rotation) keys in CPU RAM instead of GPU VRAM.

        With `on_host=True` each key is streamed GPU-side only while a rotation
        uses it: far less VRAM (one key resident, not all of them), but a
        host->device copy per rotation. Needed to fit bootstrapping keys on
        small GPUs. Defaults to device storage (faster) when never called.
        """
        self._ensure_not_built()
        self._galois_keys_on_host = on_host
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

        # All but the last entry are Q (usable levels); the last is the P
        # prime size. METHOD_II keyswitching-key size scales with
        # dnum ~ sum(Q)/sum(P), so we pick enough same-size P primes to keep
        # dnum ~ 8 (and >= 2, METHOD_II's minimum). A long bootstrapping chain
        # thus gets ~3 P primes; a short one gets 2.
        q_bits = self._coeff_modulus_bit_sizes[:-1]
        p_size = self._coeff_modulus_bit_sizes[-1]
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
        self._keygen.generate_galois_key(
            self._gk, self._sk, self._galois_keys_on_host
        )

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

    def _network_shifts(self) -> List[int]:
        # Every rotation the SDK performs (matmul rotate-by-1, _sum_slots
        # halving steps) is a power of 2, so this fixed set covers all layers.
        slot_count = self._poly_modulus_degree // 2
        return [2**k for k in range(int(math.log2(slot_count)))]

    def _usable_levels(self) -> int:
        """Multiplication levels available in a freshly encrypted ciphertext."""
        return self.encrypt([0.0])._ct.level

    def _setup_bootstrapping(self) -> None:
        """Precompute SLIM params and extend the Galois key with boot shifts.

        Idempotent: a second call is a no-op. Merges the bootstrapping BSGS
        shifts (non-powers-of-2) with the network's power-of-2 shifts into one
        key, since the bootstrap circuit needs exact keys for its own rotations.
        """
        if self._bootstrapping_ready:
            return
        # less_key_mode=True: trades extra circuit depth for ~30% fewer Galois
        # keys — required to fit bootstrapping key material on smaller GPUs.
        config = BootstrappingConfig(
            self._ctos_piece, self._stoc_piece, self._taylor_number, True
        )
        self._ops.generate_bootstrapping_params(
            self._scale, config, BootstrappingType.SLIM
        )
        boot_shifts = self._ops.bootstrapping_key_indexs()
        all_shifts = sorted(set(self._network_shifts()) | set(boot_shifts))
        gk = CKKSGaloiskey(self._backend_ctx, all_shifts)
        self._keygen.generate_galois_key(gk, self._sk, self._galois_keys_on_host)
        self._gk = gk
        self._bootstrapping_ready = True

    def _usable_after_boot(self) -> int:
        """Levels available right after a SLIM bootstrap (measured, not guessed)."""
        return self._bootstrap(self.encrypt([0.0]))._ct.level

    def _bootstrap(self, ct: EncryptedVector) -> EncryptedVector:
        """Refresh `ct` to near-full depth via SLIM bootstrapping (new ciphertext).

        SLIM runs SlotToCoeff first, so the input must sit at exactly
        `stoc_piece` levels — `ct` must arrive with at least that many.
        """
        if not self._bootstrapping_ready:
            raise RuntimeError("_setup_bootstrapping() must run before _bootstrap().")
        if ct.level < self._stoc_piece:
            raise RuntimeError(
                f"Ciphertext at level {ct.level} is below the SLIM bootstrap "
                f"input requirement ({self._stoc_piece}) — a refresh was needed "
                "earlier. Reduce activation depth (smaller ReLU `degrees`)."
            )
        raw = ct._ct.copy()
        while raw.level > self._stoc_piece:
            self._ops.mod_drop_inplace(raw)
        refreshed = self._ops.slim_bootstrapping(raw, self._gk, self._rk)
        return EncryptedVector(self, refreshed, ct._n_values)

    def _prepare_for(self, ct: EncryptedVector, needed: int) -> EncryptedVector:
        """Return a ciphertext with enough levels to run an op of depth `needed`.

        When bootstrapping is active (set up by `Sequential.compile` for deep
        networks) this refreshes the ciphertext before it would run out — and
        may fire mid-layer. A no-op when bootstrapping is inactive: there the
        network is known to fit the level budget.

        Keeps `stoc_piece` levels in hand after the upcoming op so a later SLIM
        bootstrap is still possible (SLIM needs that many input levels).
        """
        if not self._bootstrapping_ready:
            return ct
        if ct.level >= needed + self._stoc_piece:
            return ct
        refreshed = self._bootstrap(ct)
        if refreshed.level < needed:
            raise RuntimeError(
                f"A SLIM bootstrap restores {refreshed.level} levels but an "
                f"operation needs {needed} — reduce activation depth or "
                "lengthen coeff_modulus_bit_sizes."
            )
        return refreshed

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
        # Replicate cyclically: replicated[k] = values[k mod n]. The cyclic-wrap
        # diagonal matmul relies on every slot carrying x[k mod n], not zeros.
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
