import os
import csv
import time

import numpy as np

from bench.mlp.model import build_network
from bench.mlp.sdk_model import to_sdk_model, build_context
from bench.shared.io import load_weights, load_inputs, results_dir
from bench.shared.measure import cuda_sync

from core._backend import device_pool_used_bytes

MB: int = 1024 ** 2
FIELDS: list[str] = ["layer_idx", "layer_name", "time_s", "mem_delta_mb", "mem_after_mb"]


def run(case_dir: str) -> None:
    data = load_inputs(case_dir)
    sample = data["x_test"].astype(np.float32)[0]
    x_calib = data["x_calib"].astype(np.float32)

    model = load_weights(build_network(), case_dir).eval()
    ctx = build_context()
    sdk_model = to_sdk_model(model).compile(ctx, x_calib)

    vec = sdk_model.input(ctx, sample.tolist()).ciphertext

    rows: list[dict] = []
    for i, layer in enumerate(sdk_model._layers):
        cuda_sync()
        t0 = time.perf_counter()
        mem_before = device_pool_used_bytes()
        vec = layer(vec)
        cuda_sync()
        dt = time.perf_counter() - t0
        mem_after = device_pool_used_bytes()
        rows.append({
            "layer_idx": i,
            "layer_name": type(layer).__name__,
            "time_s": dt,
            "mem_delta_mb": (mem_after - mem_before) / MB,
            "mem_after_mb": mem_after / MB,
        })

    out = os.path.join(results_dir(case_dir), f"profile_sdk_{os.path.basename(case_dir)}.csv")
    with open(out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    for r in rows:
        print(f"  [{r['layer_idx']}] {r['layer_name']:14s} "
              f"{r['time_s'] * 1000:8.2f} ms   "
              f"d_mem {r['mem_delta_mb']:+8.2f} MB   live {r['mem_after_mb']:8.2f} MB")
    print("saved", out)
