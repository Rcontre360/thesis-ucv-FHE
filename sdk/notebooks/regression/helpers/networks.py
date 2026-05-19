"""Regression networks, defined with PyTorch's `nn.Sequential`.

Both train with real ReLU. The CKKS SDK runs a polynomial approximation of
ReLU instead; `Sequential.compile(calibration_data=...)` folds per-neuron
activation ranges into the Linear layers so the polynomial only ever sees
inputs normalized to about [-1, 1].
"""

from torch import nn

DEEP_BLOCKS = 20


def build_shallow(n_features: int) -> nn.Sequential:
    """3-Linear ReLU MLP. Multiplicative depth 7 under the SDK's polynomial
    ReLU — fits a normal CKKS chain, no bootstrapping needed."""
    return nn.Sequential(
        nn.Linear(n_features, 128),
        nn.ReLU(),
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Linear(64, 1),
    )


def build_deep(n_features: int) -> nn.Sequential:
    """Deep ReLU MLP. Multiplicative depth far exceeds the CKKS level budget,
    forcing repeated bootstrapping on the SDK side."""
    layers = [nn.Linear(n_features, 32), nn.ReLU()]
    for _ in range(DEEP_BLOCKS):
        layers += [nn.Linear(32, 32), nn.ReLU()]
    layers.append(nn.Linear(32, 1))
    return nn.Sequential(*layers)
