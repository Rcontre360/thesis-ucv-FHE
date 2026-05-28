import numpy as np
import torch

from bench.shared.config import SEED
from bench.playground.model import N_CLASSES, build_network
from bench.shared.io import emit, load_weights, load_inputs
from bench.shared.measure import Measure, phase_metrics
from bench.shared.metrics import accuracy, fidelity

from concrete.ml.torch.compile import compile_torch_model
import concrete.compiler

N_BITS: int = 6


def run(case_dir: str) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available")
    if not concrete.compiler.check_gpu_enabled():
        raise RuntimeError("concrete-python built without GPU support")
    if not concrete.compiler.check_gpu_available():
        raise RuntimeError("Concrete-ML cannot detect the GPU")

    data = load_inputs(case_dir)
    x_test = data["x_test"].astype(np.float32)
    y_test = data["y_test"]
    float_logits = data["float_logits"]
    calib = torch.tensor(data["x_calib"], dtype=torch.float32)
    n_test = len(x_test)

    model = load_weights(build_network(), case_dir).cpu().eval()
    enc_logits = np.empty((n_test, N_CLASSES), dtype=np.float64)

    with Measure() as m_compile:
        cml_model = compile_torch_model(
            model, torch_inputset=calib, n_bits=N_BITS, device="cuda",
        )

    with Measure() as m_keygen:
        cml_model.fhe_circuit.keygen(force=True, seed=SEED)

    with Measure() as m_infer:
        for i, x in enumerate(x_test):
            enc_logits[i] = cml_model.forward(x.reshape(1, -1), fhe="execute").reshape(-1)[:N_CLASSES]

    try:
        approx_logits = np.stack([
            cml_model.forward(x.reshape(1, -1), fhe="disable").reshape(-1)[:N_CLASSES]
            for x in x_test
        ])
        approx_accuracy = accuracy(approx_logits.argmax(axis=1), y_test)
    except Exception:
        approx_accuracy = None
    agreement, output_mae, precision = fidelity(float_logits, enc_logits)

    emit({
        "backend": "concrete-ml",
        "float_accuracy": accuracy(float_logits.argmax(axis=1), y_test),
        "approx_accuracy": approx_accuracy,
        "accuracy": accuracy(enc_logits.argmax(axis=1), y_test),
        "agreement": agreement,
        "output_mae": output_mae,
        "precision_bits": precision,

        "keygen_s": m_keygen.elapsed_s,
        "compile_s": m_compile.elapsed_s,
        "latency_s": m_infer.elapsed_s / n_test,

        **phase_metrics({"compile": m_compile, "keygen": m_keygen, "infer": m_infer}),
    })
