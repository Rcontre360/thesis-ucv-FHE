import numpy as np


def accuracy(pred, true):
    pred = np.asarray(pred)
    true = np.asarray(true)
    return float((pred == true).mean())


def mae(a, b):
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    return float(np.mean(np.abs(a - b)))


def precision_bits(reference, approx):
    err = mae(reference, approx)
    if err <= 0.0:
        return float("inf")
    return float(-np.log2(err))


def fidelity(reference_logits, logits):
    """Faithfulness of `logits` to a cleartext-model reference.

    Returns (top1_agreement, output_mae, precision_bits_or_None): the fraction
    of samples whose argmax matches the reference, the mean abs logit error,
    and -log2(error) in bits (None when the error is 0).
    """
    ref = np.asarray(reference_logits, dtype=np.float64)
    out = np.asarray(logits, dtype=np.float64)
    agree = accuracy(out.argmax(axis=1), ref.argmax(axis=1))
    err = mae(ref, out)
    bits = None if err <= 0.0 else float(-np.log2(err))
    return agree, err, bits
