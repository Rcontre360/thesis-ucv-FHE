import torch

from api import FHEContext
from api.layers.linear import Linear
from api.functions.relu import ReLU
from api.sequential import Sequential
from core.enums import SecurityLevel

# Same config as notebooks/shallow_mlp.ipynb. POLY_DEGREE=65536 with this modulus
# chain enables bootstrapping but requires ~24 GB VRAM for the galois key set.
POLY_DEGREE: int = 65_536
COEFF_MODULUS: list[int] = [60] + [52] * 28 + [60]
SCALE: int = 2 ** 52
SECURITY = SecurityLevel.SEC128
RELU_DEGREES: tuple[int, ...] = (5,) * 12


def build_context() -> FHEContext:
    return (FHEContext()
        .set_poly_modulus_degree(POLY_DEGREE)
        .set_coeff_modulus_bit_sizes(COEFF_MODULUS)
        .set_scale(SCALE)
        .set_security_level(SECURITY)
        .build())


def to_sdk_model(model: torch.nn.Module) -> Sequential:
    layers: list = []
    for m in model.cpu():
        if isinstance(m, torch.nn.Linear):
            layers.append(Linear(
                m.in_features, m.out_features,
                m.weight.detach().numpy(), m.bias.detach().numpy(),
            ))
        elif isinstance(m, torch.nn.ReLU):
            layers.append(ReLU(degrees=RELU_DEGREES))
        else:
            raise TypeError(type(m).__name__)
    return Sequential(layers)
