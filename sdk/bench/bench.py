import os
import sys
import time
import importlib
import subprocess
import tempfile

import pynvml
import pandas as pd

from bench.shared.config import (
    case_dir, interpreter_for, samples_for,
    ENV_VRAM_BASELINE, ENV_RESULT_FILE, ENV_LATENCY_N, ENV_ACCURACY_N, ENV_BENCH_CASE,
)
from bench.shared.io import read_result, results_dir

TRAIN: str = "train"
BACKENDS: list[str] = ["run_pytorch", "run_sdk", "run_cml", "run_orion"]
PROFILE: str = "profile_sdk"


def run_process(case: str, proc: str) -> None:
    module = importlib.import_module(f"bench.{case}.processes.{proc}")
    module.run(case_dir(case))


def _gpu_baseline_bytes() -> int:
    try:
        pynvml.nvmlInit()
        return int(pynvml.nvmlDeviceGetMemoryInfo(pynvml.nvmlDeviceGetHandleByIndex(0)).used)
    except Exception:
        return 0


def _interpreter(proc: str) -> str:
    bid = proc[len("run_"):] if proc.startswith("run_") else proc
    return interpreter_for(bid)


def _run(case: str, proc: str, env: dict[str, str]) -> int:
    cmd = [_interpreter(proc), "-m", "bench", case, proc]
    return subprocess.run(cmd, env=env).returncode


def _run_backend(case: str, proc: str, env: dict[str, str]) -> dict | None:
    python = _interpreter(proc)
    print(f"[bench] running {proc} [{python}] ...", flush=True)
    with tempfile.TemporaryDirectory() as tmp:
        result_file = os.path.join(tmp, "result.json")
        result = subprocess.run(
            [python, "-m", "bench", case, proc],
            text=True, capture_output=True,
            env=dict(env, **{ENV_RESULT_FILE: result_file}),
        )
        if result.returncode != 0:
            print(f"[bench] SKIP {proc}: rc={result.returncode}\n{result.stderr}", file=sys.stderr, flush=True)
            return None
        row = read_result(result_file)
        if row is None:
            print(f"[bench] SKIP {proc}: no result\n{result.stdout}", file=sys.stderr, flush=True)
        return row


def orchestrate(case: str) -> None:
    env: dict[str, str] = dict(os.environ, **{
        ENV_VRAM_BASELINE: str(_gpu_baseline_bytes()),
        ENV_BENCH_CASE: case,
    })

    if _run(case, TRAIN, env) != 0:
        raise RuntimeError("train failed")

    rows: list[dict] = []
    for proc in BACKENDS:
        row = _run_backend(case, proc, env)
        if row is not None:
            rows.append(row)

    df = pd.DataFrame(rows)
    if not df.empty:
        df["setup_s"] = df["keygen_s"] + df["compile_s"]
        df = df.drop(columns=["accuracy_per_sample_s"], errors="ignore")

    out = os.path.join(results_dir(case_dir(case)), f"results_{case}.csv")
    df.to_csv(out, index=False)

    print()
    print(df.to_string(index=False))
    print("saved", out)

    _run(case, PROFILE, env)


def duration(case: str) -> None:
    env: dict[str, str] = dict(os.environ, **{
        ENV_VRAM_BASELINE: str(_gpu_baseline_bytes()),
        ENV_BENCH_CASE: case,
        ENV_LATENCY_N: "1",
        ENV_ACCURACY_N: "1",
    })

    t0 = time.perf_counter()
    if _run(case, TRAIN, env) != 0:
        raise RuntimeError("train failed")
    train_s = time.perf_counter() - t0

    rows: list[dict] = []
    for proc in BACKENDS:
        row = _run_backend(case, proc, env)
        if row is None:
            continue
        counts = samples_for(proc[len("run_"):], case)
        setup_s = row["keygen_s"] + row["compile_s"]
        per_lat = row["latency_s"]
        per_acc = row.get("accuracy_per_sample_s", per_lat)
        rows.append({
            "backend": row["backend"],
            "setup_s": setup_s,
            "per_latency_s": per_lat,
            "per_accuracy_s": per_acc,
            "latency_n": counts.latency,
            "accuracy_n": counts.accuracy,
            "full_latency_s": counts.latency * per_lat,
            "full_accuracy_s": counts.accuracy * per_acc,
            "total_s": setup_s + counts.latency * per_lat + counts.accuracy * per_acc,
        })

    df = pd.DataFrame(rows)
    total_s = train_s + (df["total_s"].sum() if not df.empty else 0.0)

    print()
    print(f"[duration] projection for '{case}' (1 sample timed; train one-shot {train_s:.2f}s)")
    if not df.empty:
        print(df.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print(f"GRAND TOTAL: {total_s:.2f}s  (~{total_s / 60:.1f} min)")
