import numpy as np


def accuracy(pred: np.ndarray, true: np.ndarray) -> float:
    pred = np.asarray(pred)
    true = np.asarray(true)
    return float((pred == true).mean())


def mae(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    return float(np.mean(np.abs(a - b)))


def precision_bits(reference: np.ndarray, approx: np.ndarray) -> float:
    err = mae(reference, approx)
    if err <= 0.0:
        return float("inf")
    return float(-np.log2(err))


def fidelity(reference_logits: np.ndarray, logits: np.ndarray) -> tuple[float, float, float | None]:
    ref = np.asarray(reference_logits, dtype=np.float64)
    out = np.asarray(logits, dtype=np.float64)
    agree = accuracy(out.argmax(axis=1), ref.argmax(axis=1))
    err = mae(ref, out)
    bits = None if err <= 0.0 else float(-np.log2(err))
    return agree, err, bits


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=np.float64).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    ss_res = float(((y_true - y_pred) ** 2).sum())
    ss_tot = float(((y_true - y_true.mean()) ** 2).sum()) + 1e-12
    return 1.0 - ss_res / ss_tot


def pred_fidelity(reference: np.ndarray, predictions: np.ndarray) -> tuple[float, float | None]:
    err = mae(reference, predictions)
    bits = None if err <= 0.0 else float(-np.log2(err))
    return err, bits
