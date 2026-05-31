import numpy as np
import torch

from bench.shared.config import SEED, resolve_samples
from bench.mlp.model import build_network
from bench.shared.io import emit, load_weights, load_inputs
from bench.shared.measure import Measure, Timer, phase_metrics
from bench.shared.metrics import r2_score, pred_fidelity

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

    counts = resolve_samples("cml")
    data = load_inputs(case_dir)
    x_test = data["x_test"].astype(np.float32)
    y_test = data["y_test"]
    float_preds = data["float_preds"]
    calib = torch.tensor(data["x_calib"], dtype=torch.float32)

    x_lat = x_test[:counts.latency]
    x_acc = x_test[:counts.accuracy]
    y_acc = y_test[:counts.accuracy]
    float_lat = float_preds[:counts.latency]

    model = load_weights(build_network(), case_dir).cpu().eval()
    enc_preds = np.empty(counts.latency, dtype=np.float64)

    with Measure() as m_compile:
        cml_model = compile_torch_model(
            model, torch_inputset=calib, n_bits=N_BITS, device="cuda",
        )

    with Measure() as m_keygen:
        cml_model.fhe_circuit.keygen(force=True, seed=SEED)

    with Measure() as m_infer:
        for i, x in enumerate(x_lat):
            enc_preds[i] = float(cml_model.forward(x.reshape(1, -1), fhe="execute").reshape(-1)[0])

    with Timer() as t_acc:
        acc_preds = np.array([
            float(cml_model.forward(x.reshape(1, -1), fhe="disable").reshape(-1)[0])
            for x in x_acc
        ])
    approx_r2 = r2_score(y_acc, acc_preds)

    output_mae, precision = pred_fidelity(float_lat, enc_preds)

    emit({
        "backend": "concrete-ml",
        "float_r2": r2_score(y_test, float_preds),
        "approx_r2": approx_r2,
        "r2": approx_r2,
        "output_mae": output_mae,
        "precision_bits": precision,

        "keygen_s": m_keygen.elapsed_s,
        "compile_s": m_compile.elapsed_s,
        "latency_s": m_infer.elapsed_s / counts.latency,
        "accuracy_per_sample_s": t_acc.elapsed_s / counts.accuracy,
        "latency_n": counts.latency,
        "accuracy_n": counts.accuracy,

        **phase_metrics({"compile": m_compile, "keygen": m_keygen, "infer": m_infer}),
    })
