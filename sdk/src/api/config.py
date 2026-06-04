from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from core.enums import SecurityLevel


_SECURITY_CAPS = {
    SecurityLevel.SEC128: {12: 109, 13: 218, 14: 438, 15: 881, 16: 1761},
    SecurityLevel.SEC192: {12: 74,  13: 149, 14: 300, 15: 605, 16: 1212},
    SecurityLevel.SEC256: {12: 57,  13: 115, 14: 232, 15: 465, 16: 930},
}
_VALID_LOG_N = {12, 13, 14, 15, 16}
_PRIME_BITS_MIN = 30
_PRIME_BITS_MAX = 60
_BOOTSTRAP_PIECE_MIN = 2
_BOOTSTRAP_PIECE_MAX = 5
_TAYLOR_MIN = 6
_TAYLOR_MAX = 15
_SLIM_BOOTSTRAP_OVERHEAD = 9


class _ValidatedDataclass:
    def __post_init__(self) -> None:
        self._validate()
        object.__setattr__(self, "_post_init_done", True)

    def __setattr__(self, name: str, value) -> None:
        if name == "_post_init_done" or not getattr(self, "_post_init_done", False):
            object.__setattr__(self, name, value)
            return
        had_attr = hasattr(self, name)
        old_value = getattr(self, name) if had_attr else None
        object.__setattr__(self, name, value)
        try:
            self._validate()
        except Exception:
            if had_attr:
                object.__setattr__(self, name, old_value)
            else:
                object.__delattr__(self, name)
            raise

    def _validate(self) -> None:
        raise NotImplementedError


@dataclass
class BootstrapConfig(_ValidatedDataclass):
    """SLIM bootstrap circuit knobs."""
    ctos_piece: int = 3
    stoc_piece: int = 3
    taylor_number: int = 11

    def _validate(self) -> None:
        self._validate_piece("ctos_piece", self.ctos_piece)
        self._validate_piece("stoc_piece", self.stoc_piece)
        self._validate_taylor_number()

    def _validate_piece(self, name: str, value: int) -> None:
        if not _BOOTSTRAP_PIECE_MIN <= value <= _BOOTSTRAP_PIECE_MAX:
            raise ValueError(
                f"{name} must be in [{_BOOTSTRAP_PIECE_MIN}, {_BOOTSTRAP_PIECE_MAX}], "
                f"got {value}"
            )

    def _validate_taylor_number(self) -> None:
        if not _TAYLOR_MIN <= self.taylor_number <= _TAYLOR_MAX:
            raise ValueError(
                f"taylor_number must be in [{_TAYLOR_MIN}, {_TAYLOR_MAX}], "
                f"got {self.taylor_number}"
            )


@dataclass
class FHEConfig(_ValidatedDataclass):
    """CKKS parameters for FHEContext. Validates on every attribute write."""
    log_n: int = 14
    coeff_modulus_bit_sizes: List[int] = field(
        default_factory=lambda: [60, 40, 40, 40, 40, 60]
    )
    log_scale: int = 40
    security_level: SecurityLevel = SecurityLevel.SEC128
    galois_keys_on_host: bool = False
    bootstrap: Optional[BootstrapConfig] = None
    relu_degrees: Tuple[int, ...] = (7,) * 12

    def _validate(self) -> None:
        self._validate_log_n()
        self._validate_coeff_modulus_shape()
        self._validate_coeff_modulus_bits()
        self._validate_log_scale()
        self._validate_bootstrap_type()
        self._validate_security_cap()
        self._validate_coefficient_validator()
        self._validate_bootstrap_chain_depth()
        self._validate_relu_degrees()

    def _validate_log_n(self) -> None:
        if self.log_n not in _VALID_LOG_N:
            raise ValueError(
                f"log_n must be one of {sorted(_VALID_LOG_N)} "
                f"(N = 2^log_n in {{4096, ..., 65536}}), got {self.log_n}"
            )

    def _validate_coeff_modulus_shape(self) -> None:
        c = self.coeff_modulus_bit_sizes
        if not isinstance(c, list) or not c:
            raise ValueError("coeff_modulus_bit_sizes must be a non-empty list")
        if len(c) < 2:
            raise ValueError(
                "coeff_modulus_bit_sizes needs at least 2 entries "
                "(one Q prime + one P prime size)"
            )

    def _validate_coeff_modulus_bits(self) -> None:
        for b in self.coeff_modulus_bit_sizes:
            if not isinstance(b, int) or not _PRIME_BITS_MIN <= b <= _PRIME_BITS_MAX:
                raise ValueError(
                    f"each coeff_modulus_bit_size must be an int in "
                    f"[{_PRIME_BITS_MIN}, {_PRIME_BITS_MAX}], got {b}"
                )

    def _validate_log_scale(self) -> None:
        if not isinstance(self.log_scale, int) or self.log_scale < 1:
            raise ValueError(f"log_scale must be a positive int, got {self.log_scale}")

    def _validate_bootstrap_type(self) -> None:
        if self.bootstrap is not None and not isinstance(self.bootstrap, BootstrapConfig):
            raise TypeError(
                f"bootstrap must be a BootstrapConfig or None, "
                f"got {type(self.bootstrap).__name__}"
            )

    def _validate_security_cap(self) -> None:
        if self.security_level == SecurityLevel.NONE:
            return
        caps = _SECURITY_CAPS.get(self.security_level)
        if caps is None:
            raise ValueError(f"unknown security_level: {self.security_level}")
        cap = caps[self.log_n]
        q_bits = self.coeff_modulus_bit_sizes[:-1]
        p_size = self.coeff_modulus_bit_sizes[-1]
        num_p = max(2, round(sum(q_bits) / (8 * p_size)))
        total_bits = sum(q_bits) + num_p * p_size
        if total_bits > cap:
            raise ValueError(
                f"total modulus bits ({total_bits}) exceed the {self.security_level.name} "
                f"cap ({cap} bits) for log_n={self.log_n} (N=2^{self.log_n}). "
                f"Lower the bit sizes, shorten the chain, raise log_n, or use "
                f"SecurityLevel.NONE (insecure)."
            )

    def _validate_coefficient_validator(self) -> None:
        q_bits = self.coeff_modulus_bit_sizes[:-1]
        p_size = self.coeff_modulus_bit_sizes[-1]
        num_p = max(2, round(sum(q_bits) / (8 * p_size)))
        total_p = num_p * p_size
        for i in range(0, len(q_bits), num_p):
            chunk = q_bits[i:i + num_p]
            if sum(chunk) > total_p:
                raise ValueError(
                    f"coefficient_validator: Q chunk {chunk} (sum={sum(chunk)}) "
                    f"exceeds total P bits ({total_p}). Raise the P prime size "
                    f"(last entry of coeff_modulus_bit_sizes) or lower the Q bit sizes."
                )

    def _validate_bootstrap_chain_depth(self) -> None:
        if self.bootstrap is None:
            return
        q_size = len(self.coeff_modulus_bit_sizes) - 1
        b = self.bootstrap
        min_q_size = b.ctos_piece + b.stoc_piece + b.taylor_number + _SLIM_BOOTSTRAP_OVERHEAD
        if q_size < min_q_size:
            raise ValueError(
                f"Q chain size ({q_size}) is too short for the bootstrap config "
                f"(ctos={b.ctos_piece}, stoc={b.stoc_piece}, taylor={b.taylor_number}): "
                f"need at least {min_q_size} Q primes. Lengthen coeff_modulus_bit_sizes "
                f"or use smaller bootstrap parameters."
            )

    def _validate_relu_degrees(self) -> None:
        if not isinstance(self.relu_degrees, tuple) or not self.relu_degrees:
            raise ValueError("relu_degrees must be a non-empty tuple of odd ints >= 3")
        for d in self.relu_degrees:
            if not isinstance(d, int) or d < 3 or d % 2 == 0:
                raise ValueError(
                    f"each relu_degrees entry must be an odd int >= 3 "
                    f"(Cheon-Kim-Kim-Lee f_n constraint), got {d}"
                )
