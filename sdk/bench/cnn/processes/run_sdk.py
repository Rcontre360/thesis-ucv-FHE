import numpy as np

from bench.shared.config import resolve_samples
from bench.cnn.model import build_network, N_CLASSES, IMAGE_SHAPE
from bench.cnn.sdk_model import to_sdk_model, build_context
from bench.shared.io import emit, load_weights, load_inputs
from bench.shared.measure import Measure, phase_metrics
from bench.shared.metrics import accuracy, fidelity

from fhe_ml.backend._backend import device_pool_used_bytes


def run(case_dir: str) -> None:
    counts = resolve_samples("sdk")
    h, w = IMAGE_SHAPE
    data = load_inputs(case_dir)
    x_test = data["x_test"].astype(np.float32)
    y_test = data["y_test"]
    x_calib = data["x_calib"].astype(np.float32)
    float_logits = data["float_logits"]

    # Single encrypted loop sized to counts.accuracy; counts.latency is 0 by
    # config because forward_plain is NOT a clear-equivalent of the encrypted
    # path (see config.toml header).
    n = counts.accuracy
    x_acc = x_test[:n]
    y_acc = y_test[:n]
    float_acc = float_logits[:n]

    model = load_weights(build_network(), case_dir).eval()
    enc_logits = np.empty((n, N_CLASSES), dtype=np.float64)

    with Measure(alloc_probe=device_pool_used_bytes) as m_keygen:
        ctx = build_context()

    with Measure(alloc_probe=device_pool_used_bytes) as m_compile:
        sdk_model = to_sdk_model(model).compile(ctx, x_calib)

    with Measure(alloc_probe=device_pool_used_bytes) as m_infer:
        for i, x in enumerate(x_acc):
            img = x.reshape(h, w)
            enc_logits[i] = sdk_model(sdk_model.input(ctx, img.tolist())).decrypt()[:N_CLASSES]

    per_sample_s = m_infer.elapsed_s / n
    enc_top1 = accuracy(enc_logits.argmax(axis=1), y_acc)
    agreement, output_mae, precision = fidelity(float_acc, enc_logits)

    emit({
        "backend": "sdk",
        "float_accuracy": accuracy(float_logits.argmax(axis=1), y_test),
        # approx_accuracy and accuracy are equal here: the same encrypted loop
        # produces both. Both columns retained for cross-backend table consistency.
        "approx_accuracy": enc_top1,
        "accuracy": enc_top1,
        "agreement": agreement,
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
