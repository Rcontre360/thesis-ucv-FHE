"""Convolutional network for MNIST.

Trains in plaintext with real ReLU; the CKKS SDK runs a polynomial ReLU and
folds calibrated per-neuron ranges into the surrounding Linear/Conv layers.

No pooling — strided convolution does the spatial downsampling (real pooling
is non-polynomial and impossible in CKKS).
"""

from torch import nn


def build_shallow_cnn() -> nn.Sequential:
    """Conv -> ReLU -> Linear -> ReLU -> Linear, multiplicative depth 101.

    Conv(1 -> 4, kernel 5, stride 2) maps 28x28 -> 4x12x12 = 576 features;
    two Linear layers project to 10 class logits. Same depth as the shallow
    regression MLP, just with Conv at the front.
    """
    return nn.Sequential(
        nn.Conv2d(in_channels=1, out_channels=4, kernel_size=5, stride=2),
        nn.ReLU(),
        nn.Flatten(),
        nn.Linear(4 * 12 * 12, 32),
        nn.ReLU(),
        nn.Linear(32, 10),
    )
