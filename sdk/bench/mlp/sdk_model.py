import torch

from fhe_ml import FHEContext, Sequential
from bench.mlp.model import N_FEATURES

SCALE = 52
RELU_DEGREES = (5,) * 12


def to_sdk_model(model: torch.nn.Module) -> Sequential:
    return Sequential.from_torch(model, input_shape=(N_FEATURES,))


def build_context(sdk_model: Sequential) -> FHEContext:
    return FHEContext(sdk_model.generate_config(SCALE, RELU_DEGREES))
