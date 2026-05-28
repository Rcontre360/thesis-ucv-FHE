import os
import sys
import importlib
import subprocess
import tempfile

import pynvml
import pandas as pd

from bench.paths import SDK_ROOT, case_dir
from bench.shared.runner import read_result
from bench.shared.envs import interpreter_for

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
    return subprocess.run(cmd, cwd=SDK_ROOT, env=env).returncode


def _run_backend(case: str, proc: str, env: dict[str, str]) -> dict | None:
    python = _interpreter(proc)
    print(f"[bench] running {proc} [{python}] ...", flush=True)
    with tempfile.TemporaryDirectory() as tmp:
        result_file = os.path.join(tmp, "result.json")
        result = subprocess.run(
            [python, "-m", "bench", case, proc],
            cwd=SDK_ROOT, text=True, capture_output=True,
            env=dict(env, BENCH_RESULT_FILE=result_file),
        )
        if result.returncode != 0:
            print(f"[bench] SKIP {proc}: rc={result.returncode}\n{result.stderr}", file=sys.stderr, flush=True)
            return None
        row = read_result(result_file)
        if row is None:
            print(f"[bench] SKIP {proc}: no result\n{result.stdout}", file=sys.stderr, flush=True)
        return row


def orchestrate(case: str) -> None:
    env: dict[str, str] = dict(os.environ, BENCH_VRAM_BASELINE_BYTES=str(_gpu_baseline_bytes()))

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
        df.insert(0, "case", case)

    out = os.path.join(case_dir(case), "results.csv")
    df.to_csv(out, index=False)

    print()
    print(df.to_string(index=False))
    print("saved", out)

    _run(case, PROFILE, env)
