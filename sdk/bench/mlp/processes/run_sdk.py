import numpy as np

from bench.shared.config import resolve_samples
from bench.mlp.model import build_network
from bench.mlp.sdk_model import to_sdk_model, build_context
from bench.shared.io import emit, load_weights, load_inputs
from bench.shared.measure import Measure, Timer, phase_metrics
from bench.shared.metrics import r2_score, pred_fidelity

from core._backend import device_pool_used_bytes


def run(case_dir: str) -> None:
    counts = resolve_samples("sdk")
    data = load_inputs(case_dir)
    x_test = data["x_test"].astype(np.float32)
    y_test = data["y_test"]
    x_calib = data["x_calib"].astype(np.float32)
    float_preds = data["float_preds"]

    x_lat = x_test[:counts.latency]
    y_lat = y_test[:counts.latency]
    x_acc = x_test[:counts.accuracy]
    y_acc = y_test[:counts.accuracy]
    float_lat = float_preds[:counts.latency]

    model = load_weights(build_network(), case_dir).eval()
    enc_preds = np.empty(counts.latency, dtype=np.float64)

    with Measure(alloc_probe=device_pool_used_bytes) as m_keygen:
        ctx = build_context()

    with Measure(alloc_probe=device_pool_used_bytes) as m_compile:
        sdk_model = to_sdk_model(model).compile(ctx, x_calib)

    with Measure(alloc_probe=device_pool_used_bytes) as m_infer:
        for i, x in enumerate(x_lat):
            enc_preds[i] = sdk_model(sdk_model.input(ctx, x.tolist())).decrypt()[0]

    with Timer() as t_acc:
        acc_preds = np.array([float(np.asarray(sdk_model.forward_plain(x))[0]) for x in x_acc])
    approx_r2 = r2_score(y_acc, acc_preds)

    output_mae, precision = pred_fidelity(float_lat, enc_preds)

    emit({
        "backend": "sdk",
        "float_r2": r2_score(y_test, float_preds),
        "approx_r2": approx_r2,
        "r2": r2_score(y_lat, enc_preds),
        "output_mae": output_mae,
        "precision_bits": precision,

        "keygen_s": m_keygen.elapsed_s,
        "compile_s": m_compile.elapsed_s,
        "latency_s": m_infer.elapsed_s / counts.latency,
        "accuracy_per_sample_s": t_acc.elapsed_s / counts.accuracy,
        "latency_n": counts.latency,
        "accuracy_n": counts.accuracy,

        **phase_metrics({"keygen": m_keygen, "compile": m_compile, "infer": m_infer}),
    })
