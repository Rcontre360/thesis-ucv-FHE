import torch

from fhe_ml import FHEConfig, FHEContext, SecurityLevel, Sequential
from bench.mlp.model import N_FEATURES


def build_context() -> FHEContext:
    config = FHEConfig(
        log_n=16,
        coeff_modulus_bit_sizes=[60] + [52] * 28 + [60],
        log_scale=52,
        security_level=SecurityLevel.SEC128,
        relu_degrees=(5,) * 12,
    )
    return FHEContext(config).build()


def to_sdk_model(model: torch.nn.Module) -> Sequential:
    return Sequential.from_torch(model, input_shape=(N_FEATURES,))
