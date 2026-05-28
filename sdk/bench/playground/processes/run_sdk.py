import numpy as np

from bench.playground.model import N_CLASSES, build_network
from bench.playground.sdk_model import to_sdk_model, build_context
from bench.shared.io import emit, load_weights, load_inputs
from bench.shared.measure import Measure, phase_metrics
from bench.shared.metrics import accuracy, fidelity

from core._backend import device_pool_used_bytes


def run(case_dir: str) -> None:
    data = load_inputs(case_dir)
    x_test = data["x_test"].astype(np.float32)
    y_test = data["y_test"]
    x_calib = data["x_calib"].astype(np.float32)
    float_logits = data["float_logits"]
    n_test = len(x_test)

    model = load_weights(build_network(), case_dir).eval()
    enc_logits = np.empty((n_test, N_CLASSES), dtype=np.float64)

    with Measure(alloc_probe=device_pool_used_bytes) as m_keygen:
        ctx = build_context()

    with Measure(alloc_probe=device_pool_used_bytes) as m_compile:
        sdk_model = to_sdk_model(model).compile(ctx, x_calib)

    with Measure(alloc_probe=device_pool_used_bytes) as m_infer:
        for i, x in enumerate(x_test):
            enc_logits[i] = sdk_model(sdk_model.input(ctx, x.tolist())).decrypt()[:N_CLASSES]

    try:
        approx_logits = np.stack([np.asarray(sdk_model.forward_plain(x))[:N_CLASSES] for x in x_test])
        approx_accuracy = accuracy(approx_logits.argmax(axis=1), y_test)
    except Exception:
        approx_accuracy = None
    agreement, output_mae, precision = fidelity(float_logits, enc_logits)

    emit({
        "backend": "sdk",
        "float_accuracy": accuracy(float_logits.argmax(axis=1), y_test),
        "approx_accuracy": approx_accuracy,
        "accuracy": accuracy(enc_logits.argmax(axis=1), y_test),
        "agreement": agreement,
        "output_mae": output_mae,
        "precision_bits": precision,

        "keygen_s": m_keygen.elapsed_s,
        "compile_s": m_compile.elapsed_s,
        "latency_s": m_infer.elapsed_s / n_test,

        **phase_metrics({"keygen": m_keygen, "compile": m_compile, "infer": m_infer}),
    })
