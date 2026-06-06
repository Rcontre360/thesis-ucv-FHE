import numpy as np

from bench.shared.config import resolve_samples
from bench.mlp.model import build_network
from bench.mlp.sdk_model import to_sdk_model, build_context
from bench.shared.io import emit, load_weights, load_inputs
from bench.shared.measure import Measure, phase_metrics
from bench.shared.metrics import r2_score, pred_fidelity

from fhe_ml.backend._backend import device_pool_used_bytes


def run(case_dir: str) -> None:
    counts = resolve_samples("sdk")
    data = load_inputs(case_dir)
    x_test = data["x_test"].astype(np.float32)
    y_test = data["y_test"]
    x_calib = data["x_calib"].astype(np.float32)
    float_preds = data["float_preds"]

    n = counts.accuracy
    x_acc = x_test[:n]
    y_acc = y_test[:n]
    float_acc = float_preds[:n]

    model = load_weights(build_network(), case_dir).eval()
    enc_preds = np.empty(n, dtype=np.float64)

    with Measure(alloc_probe=device_pool_used_bytes) as m_keygen:
        ctx = build_context()

    with Measure(alloc_probe=device_pool_used_bytes) as m_compile:
        sdk_model = to_sdk_model(model).compile(ctx, x_calib)

    with Measure(alloc_probe=device_pool_used_bytes) as m_infer:
        for i, x in enumerate(x_acc):
            enc_preds[i] = sdk_model(sdk_model.input(ctx, x.tolist())).decrypt()[0]

    per_sample_s = m_infer.elapsed_s / n
    encrypted_r2 = r2_score(y_acc, enc_preds)
    output_mae, precision = pred_fidelity(float_acc, enc_preds)

    emit({
        "backend": "sdk",
        "float_r2": r2_score(y_test, float_preds),
        # approx_r2 and r2 are equal here: the same encrypted loop produces
        # both. Both columns retained for cross-backend table consistency.
        "approx_r2": encrypted_r2,
        "r2": encrypted_r2,
        "output_mae": output_mae,
        "precision_bits": precision,

        "keygen_s": m_keygen.elapsed_s,
        "compile_s": m_compile.elapsed_s,
        "latency_s": per_sample_s,
        "accuracy_per_sample_s": per_sample_s,
        "latency_n": counts.latency,  # 0 — no separate latency loop, see config
        "accuracy_n": counts.accuracy,

        **phase_metrics({"keygen": m_keygen, "compile": m_compile, "infer": m_infer}),
    })
