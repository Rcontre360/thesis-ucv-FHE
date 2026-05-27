import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shared.runner import run_step, collect
from shared.envs import backend_id, interpreter_for

CASE_DIR = os.path.dirname(os.path.abspath(__file__))
CASE = os.path.basename(CASE_DIR)

BACKENDS = ["run_pytorch.py", "run_sdk.py", "run_cml.py", "run_orion.py"]


def gpu_baseline_bytes():
    try:
        import pynvml
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        return int(pynvml.nvmlDeviceGetMemoryInfo(handle).used)
    except Exception:
        return 0


def main():
    baseline = gpu_baseline_bytes()
    env = dict(os.environ, BENCH_VRAM_BASELINE_BYTES=str(baseline))
    print(f"[bench] gpu vram baseline: {baseline / 1024 ** 2:.1f} MB", flush=True)

    run_step(os.path.join(CASE_DIR, "train.py"), env=env)

    scripts = []
    interpreters = {}
    for name in BACKENDS:
        try:
            python = interpreter_for(backend_id(name))
        except FileNotFoundError as exc:
            print(f"[bench] SKIP {name}: {exc}", file=sys.stderr, flush=True)
            continue
        scripts.append(os.path.join(CASE_DIR, name))
        if python:
            interpreters[name] = python

    df = collect(scripts, interpreters=interpreters, env=env)
    if not df.empty:
        df["setup_s"] = df["keygen_s"] + df["compile_s"]
    df.insert(0, "case", CASE)
    out = os.path.join(CASE_DIR, "results.csv")
    df.to_csv(out, index=False)
    print()
    print(df.to_string(index=False))
    print("saved", out)

    # SDK-only metrics (not part of the cross-library comparison) run last.
    profile = os.path.join(CASE_DIR, "profile_sdk.py")
    if os.path.exists(profile):
        try:
            run_step(profile, python=interpreter_for("sdk"), env=env)
        except Exception as exc:
            print(f"[bench] SKIP profile_sdk.py: {exc}", file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()
