import torch

from opal_ml import FHEContext, Sequential
from bench.cnn.model import CHANNELS, IMAGE_SHAPE

SCALE = 53
RELU_DEGREES = (5,) * 6  # depth 27 -> fits at scale 53 with NO bootstrap (was (5,)*10)


def to_sdk_model(model: torch.nn.Module) -> Sequential:
    return Sequential.from_torch(model, input_shape=(CHANNELS, *IMAGE_SHAPE))


def build_context(sdk_model: Sequential) -> FHEContext:
    config = sdk_model.generate_config(SCALE, RELU_DEGREES)
    config.galois_keys_on_host = True
    return FHEContext(config)
