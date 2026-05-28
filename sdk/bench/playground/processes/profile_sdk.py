import os
import csv
import time

import numpy as np
import torch

from bench.playground.model import build_network
from bench.playground.sdk_model import to_sdk_model, build_context
from bench.shared.io import load_weights, load_inputs

from core._backend import device_pool_used_bytes

MB: int = 1024 ** 2
FIELDS: list[str] = ["layer_idx", "layer_name", "time_s", "mem_delta_mb", "mem_after_mb"]


def _sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def run(case_dir: str) -> None:
    sample = load_inputs(case_dir)["x_test"].astype(np.float32)[0]

    model = load_weights(build_network(), case_dir).eval()
    ctx = build_context()
    sdk_model = to_sdk_model(model).compile(ctx)

    vec = sdk_model.input(ctx, sample.tolist()).ciphertext

    rows: list[dict] = []
    for i, layer in enumerate(sdk_model._layers):
        _sync()
        t0 = time.perf_counter()
        mem_before = device_pool_used_bytes()
        vec = layer(vec)
        _sync()
        dt = time.perf_counter() - t0
        mem_after = device_pool_used_bytes()
        rows.append({
            "layer_idx": i,
            "layer_name": type(layer).__name__,
            "time_s": dt,
            "mem_delta_mb": (mem_after - mem_before) / MB,
            "mem_after_mb": mem_after / MB,
        })

    out = os.path.join(case_dir, "profile_sdk.csv")
    with open(out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    for r in rows:
        print(f"  [{r['layer_idx']}] {r['layer_name']:14s} "
              f"{r['time_s'] * 1000:8.2f} ms   "
              f"d_mem {r['mem_delta_mb']:+8.2f} MB   live {r['mem_after_mb']:8.2f} MB")
    print("saved", out)
