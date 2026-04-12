from typing import TYPE_CHECKING, List, Optional, Union
import math

if TYPE_CHECKING:
    from fhe_sdk.encrypted_vector import EncryptedVector

class Module:
    def forward(self, x: "EncryptedVector") -> "EncryptedVector":
        raise NotImplementedError("Each layer must implement forward()")

    def __call__(self, x: "EncryptedVector") -> "EncryptedVector":
        return self.forward(x)

class Sequential(Module):
    def __init__(self, *layers: Module) -> None:
        self._layers = list(layers)

    def forward(self, x: "EncryptedVector") -> "EncryptedVector":
        out = x
        for layer in self._layers:
            out = layer.forward(out)
        return out

    def __getitem__(self, index: int) -> Module:
        return self._layers[index]

    def __len__(self) -> int:
        return len(self._layers)

class Linear(Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True) -> None:
        self.in_features = in_features
        self.out_features = out_features
        self.use_bias = bias
        self.weight: Optional[List[List[float]]] = None
        self.bias: Optional[List[float]] = None

    def load_weights(self, weight: Union[List[List[float]], any], bias: Optional[Union[List[float], any]] = None) -> None:
        # Simple converter for numpy/torch
        if hasattr(weight, "tolist"): weight = weight.tolist()
        if hasattr(bias, "tolist"): bias = bias.tolist()
        
        if len(weight) != self.out_features or len(weight[0]) != self.in_features:
            raise ValueError(f"Weight shape mismatch. Expected ({self.out_features}, {self.in_features})")
        
        self.weight = weight
        if self.use_bias:
            if bias is None: raise ValueError("Bias is required when use_bias=True")
            self.bias = bias

    def forward(self, x: "EncryptedVector") -> "EncryptedVector":
        if self.weight is None:
            raise RuntimeError("Weights not loaded. Call load_weights() first.")
        
        # Matrix-vector multiplication using __rmatmul__ (diagonal method)
        # Note: In our current implementation, we'll use a slot-wise dot product 
        # or wait for the optimized diagonal kernel. 
        # Per API.md: W @ ct
        out = self.weight @ x
        if self.use_bias:
            out = out + self.bias
        return out

class Square(Module):
    def forward(self, x: "EncryptedVector") -> "EncryptedVector":
        return x * x

class ApproxReLU(Module):
    def __init__(self, degree: int = 3, bound: float = 5.0) -> None:
        if degree % 2 == 0 or degree < 3:
            raise ValueError("Degree must be odd and >= 3")
        self.degree = degree
        self.bound = bound
        # In a real implementation, we'd pre-calculate minimax coefficients here.
        # For the prototype, we use a simple Taylor or fixed coefficients.
        if degree == 3:
            # Approx ReLU: 0.5x + 0.15x^2 ... (simplified)
            # Actually, standard quadratic approx: 0.125x^2 + 0.5x + 0.25
            self.coeffs = [0.25, 0.5, 0.125] 

    def forward(self, x: "EncryptedVector") -> "EncryptedVector":
        # Horner's method or Baby-step Giant-step for polynomial evaluation
        # Simplified evaluation: c0 + c1*x + c2*x^2
        x2 = x * x
        return (x * self.coeffs[1]) + (x2 * self.coeffs[2]) + self.coeffs[0]

class ApproxSigmoid(Module):
    def __init__(self, degree: int = 3, bound: float = 5.0) -> None:
        self.degree = degree
        self.bound = bound
        # Standard Taylor for Sigmoid at 0: 0.5 + 0.25x - 0.0208x^3
        self.coeffs = {0: 0.5, 1: 0.25, 3: -0.0208}

    def forward(self, x: "EncryptedVector") -> "EncryptedVector":
        x2 = x * x
        x3 = x2 * x
        return (x * self.coeffs[1]) + (x3 * self.coeffs[3]) + self.coeffs[0]
