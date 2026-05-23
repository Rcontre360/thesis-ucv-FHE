# ReLU Approximation in CKKS

CKKS can only add and multiply — i.e. evaluate **polynomials**. ReLU
(`max(0, x)`) is not a polynomial (it has a kink at 0), so it must be
approximated. This document records what was found while building the
regression benchmark: what fails, why, and what works.

## A single polynomial does not work

The SDK shipped a degree-2 ReLU: `0.125x² + 0.5x + 0.375`. On a 2-hidden-layer
MLP it produces **R² ≈ −90000** (real ReLU: 0.79). Two failure modes, both fatal:

**Low degree → systematic bias.** `0.125x²+0.5x+0.375` is `≥ ReLU` everywhere on
`[−1,1]` (e.g. `poly(0)=0.375` vs `ReLU(0)=0`). A ReLU-trained network has ~half
its neurons negative; the polynomial turns those exact 0s into 0.375s. The bias
is one-signed, so it does not cancel across neurons — it sums and compounds
through layers.

**High degree → divergence.** A higher-degree fit is more accurate *inside* its
domain but explodes *outside* it. Degree sweep (least-squares fit on `[−1,1]`,
2-layer net):

| degree | fit error on [−1,1] | network R² |
|---|---|---|
| 2 | 0.094 | −6,572 |
| 4 | 0.058 | −36,815 |
| 8 | 0.033 | −10³⁰ |
| 12 | 0.023 | −10¹¹⁰ |

The polynomial only resembles ReLU in a thin band at `[−1,1]`: shrink the inputs
and it collapses toward its constant term; grow them and it diverges. No
rescaling rescues a single polynomial.

## The domain problem

A polynomial approximates ReLU **only inside the interval it was fit on**.
Outside, it is a different function — and a high-degree one diverges violently.
So the activation's inputs must be kept inside the fit domain. They are not
naturally bounded, so they have to be rescaled into it.

## Calibration

Measure where the inputs actually go, then rescale:

1. Run representative (training) data through the network with **real ReLU**;
   record, per activation layer, `B = max|input|` — one scalar per layer.
2. **Fold** `B` into the surrounding Linear layers: the Linear *before* the
   activation has its weights and bias divided by `B`; the Linear *after* has
   its weights multiplied by `B`. ReLU is positively homogeneous
   (`ReLU(Bx) = B·ReLU(x)`), so the fold is **exact** — the network is
   unchanged, but the activation now sees inputs in `~[−1,1]`.

This is post-training-quantization-style calibration. Orion does the same
(`orion.fit(net, data)`, a `torch.fx` range tracker).

### Two gaps calibration cannot close

1. **Train vs test** — `B` is a training-set maximum; test samples can exceed
   it. Measured: a first-layer test input reached 2.6 against a calibrated
   bound of 1.0.
2. **Real ReLU vs the polynomial** — calibration measures the *real-ReLU*
   network's ranges, but inference runs the *polynomial*. The polynomial
   perturbs the computation, so from the 2nd activation onward the actual
   inputs drift away from the calibrated ranges. Measured: a layer-2 input
   drifted to −11.7 while calibration said `[−1,1]`. This gap dominates.

## The margin

Fold by `margin · B` instead of `B`, giving the polynomial headroom against
both gaps. A margin alone does **not** fix a bad polynomial — pairing margin 10
with the degree-2 poly squashed all inputs near 0, where that poly is just its
constant `0.375`. The margin is necessary but only useful alongside a
polynomial that behaves well across a range.

## The working approach: composite polynomial

`ReLU(x) = ½·x·(1 + sign(x))`. The hard part is `sign(x)`.

Do **not** fit `sign` with a single polynomial (Gibbs oscillation, divergence).
Instead **compose** a fixed odd polynomial `f`:

```
sign(x) ≈ f ∘ f ∘ … ∘ f (x)
```

Each `f` maps `[−1,1] → [−1,1]` and has slope > 1 at 0, so it pushes values
toward ±1; composing it converges to `sign`. Because every `f` stays within
`[−1,1]`, the composition is **bounded and stable** — unlike a single
polynomial it cannot explode on a small domain overrun.

`f` (Cheon et al. 2019, degree 7): `f(x) = (35x − 35x³ + 21x⁵ − 5x⁷)/16`.

### Precision — how many compositions

The number of compositions is what determines accuracy. Measured (2-hidden-layer
MLP, per-layer calibration, plaintext):

| compositions of `f` | network R² |
|---|---|
| 3 | −5 to −7 |
| 5 | −3.5 to −4 |
| 8 | −1.0 to 0.18 |
| **12** | **0.787 – 0.792** |

At **12 compositions** the polynomial ReLU recovers the real-ReLU network
(R² 0.792 vs 0.794). Fewer is not enough. A margin of 1.5 was marginally best.

### The cost — depth

12 compositions of a degree-7 polynomial is roughly **36+ multiplicative
levels** for one ReLU. In plaintext that is free; **under encryption it forces
bootstrapping**. Accurate ReLU in CKKS is inherently depth-heavy — even
"shallow" networks need bootstrapping.

## The minimax composite (fewer multiplications)

The `fₙ` composite needs 12 steps because `f` is a fixed, sub-optimal
polynomial. The literature (Lee et al.; used by Orion) replaces each step with a
**minimax (Remez) polynomial** of optimized degree. Orion's ReLU composes **3**
minimax polynomials of degrees **[15, 15, 27]** — the same accuracy as ~12 `fₙ`
steps at a fraction of the multiplicative depth. Orion's final polynomial
targets the 0/1 step directly, giving `ReLU(x) = x · step(x)`.

| | `fₙ` composite | minimax composite |
|---|---|---|
| polynomial per step | fixed degree-7 | minimax, optimized degree |
| steps for full accuracy | ~12 | 3 (degrees 15, 15, 27) |
| multiplicative depth | very high | much lower |
| accuracy | recovers real ReLU | recovers real ReLU |

The minimax coefficients come from a Remez routine (Orion delegates to Lattigo,
in Go). Same composite *structure*, optimized degrees. For a plaintext prototype
the `fₙ` composite is fine; for encrypted inference the minimax composite is
needed to keep the depth — hence the number of bootstraps — manageable.

## Summary

- A single polynomial cannot approximate ReLU usefully for multi-layer
  networks — low degree is biased, high degree diverges.
- The activation input must be brought into the polynomial's domain via
  per-layer **calibration** + a **margin**.
- Calibration cannot fully close the train/test and real-vs-polynomial gaps;
  the margin absorbs the residual.
- The working activation is a **composite**: `ReLU = ½x(1+sign(x))` with `sign`
  approximated by composing an odd polynomial — bounded and stable.
- **Precision** needs enough compositions: ~12 of the degree-7 `fₙ`, or 3
  minimax polynomials of degrees `[15, 15, 27]`.
- Accurate ReLU is **depth-heavy → requires bootstrapping**.

## Status

Validated in plaintext (`notebooks/regression/`): composite `fₙ` ReLU +
per-layer calibration + margin 1.5 → R² 0.792 (real ReLU 0.794).

- `helpers/poly_relu.py` — composite-`fₙ` ReLU.
- `helpers/calibration.py` — per-layer calibration, margin, weight folding.

Next: port to `src/` (the SDK `ReLU` and the calibration path); for the
encrypted path, use the minimax composite to limit multiplicative depth.

## References

- `research/ml-ckks.pdf` — Lee et al., *Privacy-Preserving ML with FHE for Deep
  Neural Network*, IEEE Access 2022 (ResNet-20; composite-minimax ReLU).
- `research/minmax-activation.pdf` — Lee et al., *Optimization of Homomorphic
  Comparison Algorithm on RNS-CKKS*, IEEE Access 2022 (minimax composite of
  the sign function).
- Orion — github.com/baahl-nyu/orion — reference implementation: calibration
  (`orion.fit`) + minimax-composite ReLU `[15,15,27]`.
- Cheon et al. 2019 — the `fₙ` sign-composite polynomials.
