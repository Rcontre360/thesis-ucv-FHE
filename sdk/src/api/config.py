"""Configuration objects for `FHEContext`.

User code constructs an `FHEConfig` (and optionally a nested `BootstrapConfig`),
overrides any fields it needs, and passes the result to `FHEContext`:

    from fhe_ml import FHEContext, FHEConfig, BootstrapConfig, SecurityLevel

    cfg = FHEConfig(
        poly_modulus_degree=65536,
        coeff_modulus_bit_sizes=[60] + [52] * 28 + [60],
        scale=2 ** 52,
        bootstrap=BootstrapConfig(taylor_number=10),
    )
    ctx = FHEContext(cfg).build()

Or, equivalently, by attribute assignment:

    cfg = FHEConfig()
    cfg.poly_modulus_degree = 65536
    cfg.bootstrap.taylor_number = 10
"""
from dataclasses import dataclass, field
from typing import List

from core.enums import SecurityLevel


@dataclass
class BootstrapConfig:
    """SLIM bootstrapping circuit knobs (used only when `Sequential.compile`
    enables bootstrapping for a deep network).

    ctos_piece, stoc_piece (2-5): DFT factor counts. Higher = deeper but
    cheaper-per-stage circuit.
    taylor_number (6-15): EvalMod sine polynomial degree. Higher = more
    accurate per refresh, but deeper.
    """
    ctos_piece: int = 3
    stoc_piece: int = 3
    taylor_number: int = 11


@dataclass
class FHEConfig:
    """CKKS parameters and key-management options for `FHEContext`.

    All fields have working defaults. Instantiate `FHEConfig()` for a working
    configuration, then override only the fields you need.
    """
    poly_modulus_degree: int = 16384
    # The last entry is the size of the P (special / keyswitch) primes; every
    # entry before it is one Q-chain prime. The Q chain length determines how
    # many multiplications can run before bootstrapping (or running out).
    coeff_modulus_bit_sizes: List[int] = field(
        default_factory=lambda: [60, 40, 40, 40, 40, 60]
    )
    scale: float = 2 ** 40
    security_level: SecurityLevel = SecurityLevel.SEC128
    # When True, Galois (rotation) keys live in CPU RAM and stream to GPU only
    # while a rotation needs them. Slower per rotation, but fits much larger
    # networks (especially bootstrapping) on smaller GPUs.
    galois_keys_on_host: bool = False
    bootstrap: BootstrapConfig = field(default_factory=BootstrapConfig)
