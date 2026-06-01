import numpy as np

from bench.shared.config import resolve_samples
from bench.cnn.model import build_network, N_CLASSES, IMAGE_SHAPE
from bench.cnn.sdk_model import to_sdk_model, build_context
from bench.shared.io import emit, load_weights, load_inputs
from bench.shared.measure import Measure, Timer, phase_metrics
from bench.shared.metrics import accuracy, fidelity

from core._backend import device_pool_used_bytes


def run(case_dir: str) -> None:
    counts = resolve_samples("sdk")
    h, w = IMAGE_SHAPE
    data = load_inputs(case_dir)
    x_test = data["x_test"].astype(np.float32)
    y_test = data["y_test"]
    x_calib = data["x_calib"].astype(np.float32)
    float_logits = data["float_logits"]

    x_lat = x_test[:counts.latency]
    x_acc = x_test[:counts.accuracy]
    y_acc = y_test[:counts.accuracy]
    float_lat = float_logits[:counts.latency]

    model = load_weights(build_network(), case_dir).eval()
    enc_logits = np.empty((counts.latency, N_CLASSES), dtype=np.float64)

    with Measure(alloc_probe=device_pool_used_bytes) as m_keygen:
        ctx = build_context()

    with Measure(alloc_probe=device_pool_used_bytes) as m_compile:
        sdk_model = to_sdk_model(model).compile(ctx, x_calib)

    with Measure(alloc_probe=device_pool_used_bytes) as m_infer:
        for i, x in enumerate(x_lat):
            img = x.reshape(h, w)
            enc_logits[i] = sdk_model(sdk_model.input(ctx, img.tolist())).decrypt()[:N_CLASSES]

    with Timer() as t_acc:
        # forward_plain is a raw `x @ W.T` and expects flat input (1·28·28=784).
        # Only the encrypted path goes through Conv2D.prepare_input which handles
        # the (28, 28) reshape.
        acc_logits = np.stack([
            np.asarray(sdk_model.forward_plain(x))[:N_CLASSES]
            for x in x_acc
        ])
    approx_accuracy = accuracy(acc_logits.argmax(axis=1), y_acc)

    agreement, output_mae, precision = fidelity(float_lat, enc_logits)

    emit({
        "backend": "sdk",
        "float_accuracy": accuracy(float_logits.argmax(axis=1), y_test),
        "approx_accuracy": approx_accuracy,
        "accuracy": accuracy(enc_logits.argmax(axis=1), y_test[:counts.latency]),
        "agreement": agreement,
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
