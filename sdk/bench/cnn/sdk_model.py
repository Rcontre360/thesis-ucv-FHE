import torch

from api import FHEContext
from api.layers.linear import Linear
from api.layers.conv2d import Conv2D
from api.functions.relu import ReLU
from api.sequential import Sequential
from core.enums import SecurityLevel
from bench.cnn.model import IMAGE_SHAPE

# Same config as notebooks/shallow_cnn.ipynb. Galois keys stored on host to fit
# the ~24 GB key set; (3,3,11) is the bootstrap factorization the notebook uses.
POLY_DEGREE: int = 65_536
COEFF_MODULUS: list[int] = [60] + [53] * 28 + [60]
SCALE: int = 2 ** 53
SECURITY = SecurityLevel.SEC128
RELU_DEGREES: tuple[int, ...] = (5,) * 9
BOOTSTRAP_PARAMS: tuple[int, int, int] = (3, 3, 11)


def build_context() -> FHEContext:
    return (FHEContext()
        .set_poly_modulus_degree(POLY_DEGREE)
        .set_coeff_modulus_bit_sizes(COEFF_MODULUS)
        .set_scale(SCALE)
        .set_security_level(SECURITY)
        .set_galois_key_storage(on_host=True)
        .set_bootstrapping_params(*BOOTSTRAP_PARAMS)
        .build())


def to_sdk_model(model: torch.nn.Module) -> Sequential:
    h, w = IMAGE_SHAPE
    layers: list = []
    for m in model.cpu():
        if isinstance(m, torch.nn.Conv2d):
            stride = m.stride[0] if isinstance(m.stride, tuple) else m.stride
            layers.append(Conv2D(
                in_channels=m.in_channels, out_channels=m.out_channels,
                kernel_size=m.kernel_size, input_shape=(h, w),
                weight=m.weight.detach().numpy(),
                bias=m.bias.detach().numpy(),
                stride=stride,
            ))
            kh, kw = m.kernel_size
            h = (h - kh) // stride + 1
            w = (w - kw) // stride + 1
        elif isinstance(m, torch.nn.Linear):
            layers.append(Linear(
                m.in_features, m.out_features,
                m.weight.detach().numpy(), m.bias.detach().numpy(),
            ))
        elif isinstance(m, torch.nn.Flatten):
            continue
        elif isinstance(m, torch.nn.ReLU):
            layers.append(ReLU(degrees=RELU_DEGREES))
        else:
            raise TypeError(type(m).__name__)
    return Sequential(layers)
