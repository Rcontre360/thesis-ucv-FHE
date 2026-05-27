import torch

from api import FHEContext
from api.layers.linear import Linear
from api.functions.square import Square as SDKSquare
from api.sequential import Sequential
from core.enums import SecurityLevel
from model import Square

POLY_DEGREE = 8192
COEFF_MODULUS = [60, 40, 40, 40, 40, 60, 60]
SCALE = 2 ** 40
SECURITY = SecurityLevel.NONE


def build_context():
    return (FHEContext()
        .set_poly_modulus_degree(POLY_DEGREE)
        .set_coeff_modulus_bit_sizes(COEFF_MODULUS)
        .set_scale(SCALE)
        .set_security_level(SECURITY)
        .build())


def to_sdk_model(model):
    layers = []
    for m in model.cpu():
        if isinstance(m, torch.nn.Linear):
            layers.append(Linear(
                m.in_features, m.out_features,
                m.weight.detach().numpy(), m.bias.detach().numpy(),
            ))
        elif isinstance(m, Square):
            layers.append(SDKSquare())
        else:
            raise TypeError(type(m).__name__)
    return Sequential(layers)
