import torch

from fhe_ml import BootstrapConfig, FHEConfig, FHEContext, SecurityLevel, Sequential
from bench.cnn.model import CHANNELS, IMAGE_SHAPE


def build_context() -> FHEContext:
    config = FHEConfig(
        log_n=16,
        coeff_modulus_bit_sizes=[60] + [53] * 27 + [60],
        log_scale=53,
        security_level=SecurityLevel.SEC128,
        galois_keys_on_host=True,
        bootstrap=BootstrapConfig(ctos_piece=3, stoc_piece=3, taylor_number=10),
        relu_degrees=(5,) * 10,
    )
    return FHEContext(config).build()


def to_sdk_model(model: torch.nn.Module) -> Sequential:
    return Sequential.from_torch(model, input_shape=(CHANNELS, *IMAGE_SHAPE))
