from dataclasses import dataclass, field
from typing import List

from core.enums import SecurityLevel


_SECURITY_CAPS = {
    SecurityLevel.SEC128: {12: 109, 13: 218, 14: 438, 15: 881, 16: 1761},
    SecurityLevel.SEC192: {12: 74,  13: 149, 14: 300, 15: 605, 16: 1212},
    SecurityLevel.SEC256: {12: 57,  13: 115, 14: 232, 15: 465, 16: 930},
}
_VALID_LOG_N = {12, 13, 14, 15, 16}
_PRIME_BITS_MIN = 14
_PRIME_BITS_MAX = 60


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
    """SLIM bootstrapping circuit knobs.

    Used only when Sequential.compile enables bootstrapping for a deep network.
    """
    ctos_piece: int = 3
    stoc_piece: int = 3
    taylor_number: int = 11

    def _validate(self) -> None:
        if not 2 <= self.ctos_piece <= 5:
            raise ValueError(f"ctos_piece must be in [2, 5], got {self.ctos_piece}")
        if not 2 <= self.stoc_piece <= 5:
            raise ValueError(f"stoc_piece must be in [2, 5], got {self.stoc_piece}")
        if not 6 <= self.taylor_number <= 15:
            raise ValueError(f"taylor_number must be in [6, 15], got {self.taylor_number}")


@dataclass
class FHEConfig(_ValidatedDataclass):
    """CKKS parameters and key-management options for FHEContext.

    Validates on construction and on every subsequent attribute assignment.
    All quantities that are exact powers of 2 are stored as their log2.
    """
    log_n: int = 14
    coeff_modulus_bit_sizes: List[int] = field(
        default_factory=lambda: [60, 40, 40, 40, 40, 60]
    )
    log_scale: int = 40
    security_level: SecurityLevel = SecurityLevel.SEC128
    galois_keys_on_host: bool = False
    bootstrap: BootstrapConfig = field(default_factory=BootstrapConfig)

    def _validate(self) -> None:
        if self.log_n not in _VALID_LOG_N:
            raise ValueError(
                f"log_n must be one of {sorted(_VALID_LOG_N)} "
                f"(N = 2^log_n in {{4096, ..., 65536}}), got {self.log_n}"
            )

        if not isinstance(self.coeff_modulus_bit_sizes, list) or not self.coeff_modulus_bit_sizes:
            raise ValueError("coeff_modulus_bit_sizes must be a non-empty list")
        if len(self.coeff_modulus_bit_sizes) < 2:
            raise ValueError(
                "coeff_modulus_bit_sizes needs at least 2 entries "
                "(one Q prime + one P prime size)"
            )
        for b in self.coeff_modulus_bit_sizes:
            if not isinstance(b, int) or not _PRIME_BITS_MIN <= b <= _PRIME_BITS_MAX:
                raise ValueError(
                    f"each coeff_modulus_bit_size must be an int in "
                    f"[{_PRIME_BITS_MIN}, {_PRIME_BITS_MAX}], got {b}"
                )

        if not isinstance(self.log_scale, int) or self.log_scale < 1:
            raise ValueError(f"log_scale must be a positive int, got {self.log_scale}")
        q_bits = self.coeff_modulus_bit_sizes[:-1]
        if self.log_scale > max(q_bits):
            raise ValueError(
                f"log_scale ({self.log_scale}) exceeds the largest Q prime "
                f"({max(q_bits)} bits); decoding precision would underflow"
            )

        if not isinstance(self.bootstrap, BootstrapConfig):
            raise TypeError(
                f"bootstrap must be a BootstrapConfig, got {type(self.bootstrap).__name__}"
            )

        if self.security_level == SecurityLevel.NONE:
            return
        caps = _SECURITY_CAPS.get(self.security_level)
        if caps is None:
            raise ValueError(f"unknown security_level: {self.security_level}")
        cap = caps[self.log_n]
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
