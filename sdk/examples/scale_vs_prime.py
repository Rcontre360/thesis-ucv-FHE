"""Does log_scale > the scaling prime make CKKS precision explode?

Chain is fixed at first=60, scaling primes=40, P=60. We sweep log_scale across
below / at / above the scaling prime (40) and up to the first prime (60), run a
short ciphertext x plaintext multiply chain, and compare the decryption to the
same computation in plaintext.

Claim under test: the scale Delta=2^log_scale must be <= the *scaling* primes
(the ones rescale divides by), NOT the first prime. If log_scale exceeds the
scaling prime, the scale grows on every rescale and the result blows up.
"""

import numpy as np

from opal_ml import Client, FHEConfig, FHEContext, SecurityLevel

FIRST = 60
SCALING = 40
LOG_N = 13
N_SCALING = 4  # usable levels
DEPTH = 3
VEC_LEN = 64
LOG_SCALES = [30, 35, 40, 45, 50, 60]  # below / at / above the scaling prime (40)


def run(log_scale: int) -> float:
    chain = [FIRST] + [SCALING] * N_SCALING + [60]
    cfg = FHEConfig(
        log_n=LOG_N,
        coeff_modulus_bit_sizes=chain,
        log_scale=log_scale,
        security_level=SecurityLevel.NONE,
    )
    ctx = FHEContext(cfg)
    client = Client(ctx)
    ctx.set_client_params(client.key_params())

    rng = np.random.default_rng(0)
    x = rng.uniform(0.3, 0.7, VEC_LEN)
    w = rng.uniform(0.85, 0.95, VEC_LEN)

    ev = client.encrypt(x.tolist())
    for _ in range(DEPTH):
        ev = ev * w.tolist()
    decoded = np.array(client.decrypt(ev)[:VEC_LEN])

    expected = x * (w**DEPTH)
    return float(np.mean(np.abs(decoded - expected)))


print(f"chain: first={FIRST}, scaling={SCALING}, depth={DEPTH} multiplies\n")
print(f"{'log_scale':>10}{'vs scaling(40)':>16}{'mean_abs_error':>18}   verdict")
for ls in LOG_SCALES:
    rel = "below" if ls < SCALING else ("same" if ls == SCALING else "ABOVE")
    try:
        err = run(ls)
        verdict = "EXPLODES" if (not np.isfinite(err) or err > 1.0) else "ok"
    except Exception as e:
        err, verdict = float("nan"), f"ERROR: {str(e)[:40]}"
    print(f"{ls:>10}{rel:>16}{err:>18.3e}   {verdict}")
