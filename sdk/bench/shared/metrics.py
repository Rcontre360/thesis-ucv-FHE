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
