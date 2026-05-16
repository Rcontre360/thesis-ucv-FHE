from typing import TYPE_CHECKING, List, Optional, Set, Union

from core.errors import LayerConfigError
from core.layer import Layer
from api.input import Input

if TYPE_CHECKING:
    from api.ciphertext import EncryptedVector
    from api.context import FHEContext


class Sequential:
    """Chain of layers and activation functions for encrypted vectors.

    Each element must be a `Layer` (defined in `core.layer`). Use
    `model.input(context, raw_data)` to validate, reshape, and encrypt raw
    user data into an `Input` in the layout this model's first layer expects.
    The model's `__call__` accepts either an `Input` or an `EncryptedVector`.

    Call `model.compile(context)` once before inference: it sizes the network
    against the context's level budget and, only if the network is too deep,
    enables SLIM bootstrapping and plans where to refresh. Models that fit the
    budget never bootstrap.

    Example:
        model = Sequential([
            Conv2D(1, 4, 3, (28, 28), conv_w, bias=conv_b),
            ReLU(),
            Linear(4 * 26 * 26, 10, lin_w),
        ])
        model.compile(context)
        inp = model.input(context, image_28x28)
        out = model(inp)
    """

    def __init__(self, layers: List[Layer]) -> None:
        if not layers:
            raise ValueError("Sequential requires at least one layer")
        for i, layer in enumerate(layers):
            if not isinstance(layer, Layer):
                raise TypeError(
                    f"layers[{i}] is {type(layer).__name__}; must inherit from `Layer`."
                )
        self._layers = list(layers)
        self._context: Optional["FHEContext"] = None
        # Layer indices to refresh the ciphertext before; empty until compile().
        self._bootstrap_before: Set[int] = set()

    def input(self, context: "FHEContext", raw_data: object) -> Input:
        """Validate, reshape, and encrypt `raw_data` via the first layer's `prepare_input`."""
        flat = self._layers[0].prepare_input(raw_data)
        return Input(context, flat)

    def compile(self, context: "FHEContext") -> "Sequential":
        """Bind the context and plan bootstrapping from each layer's `mult_depth`.

        If total depth fits the context's level budget, bootstrapping stays
        off entirely. Otherwise SLIM bootstrapping is set up and refreshes are
        scheduled at layer boundaries where the next layer would overflow.
        """
        self._context = context
        usable = context._usable_levels()

        if sum(l.mult_depth() for l in self._layers) <= usable:
            self._bootstrap_before = set()
            return self

        context._setup_bootstrapping()
        after_boot = context._usable_after_boot()
        stoc = context._stoc_piece
        deepest = max(l.mult_depth() for l in self._layers)
        # SLIM consumes `stoc` input levels, so a refreshed ciphertext must hold
        # the deepest layer AND still leave `stoc` levels for the next refresh.
        if after_boot < deepest + stoc:
            raise LayerConfigError(
                f"A bootstrap leaves {after_boot} levels, but a layer needs "
                f"{deepest} and {stoc} more must remain to bootstrap again — "
                f"lengthen coeff_modulus_bit_sizes."
            )

        schedule: Set[int] = set()
        remaining = usable
        last = len(self._layers) - 1
        for i, layer in enumerate(self._layers):
            depth = layer.mult_depth()
            # Refresh before layer i if it can't run, or if running it would
            # drop levels below `stoc` while layers still remain (no way back).
            if remaining < depth or (i < last and remaining - depth < stoc):
                schedule.add(i)
                remaining = after_boot
            remaining -= depth
        self._bootstrap_before = schedule
        return self

    def __call__(self, x: Union[Input, "EncryptedVector"]) -> "EncryptedVector":
        if isinstance(x, Input):
            x = x.ciphertext
        for i, layer in enumerate(self._layers):
            if i in self._bootstrap_before:
                x = self._context._bootstrap(x)
            x = layer(x)
        return x
