import os
import sys
import json
import subprocess

_SENTINEL = "__BENCH_RESULT__"


def emit(result):
    print(_SENTINEL + " " + json.dumps(result))
    sys.stdout.flush()


def run_step(script_path, python=None, env=None):
    python = python or sys.executable
    proc = subprocess.run([python, script_path], env=env)
    if proc.returncode != 0:
        raise RuntimeError(f"{os.path.basename(script_path)} failed (code {proc.returncode})")


def run_backend(script_path, python=None, env=None):
    python = python or sys.executable
    proc = subprocess.run(
        [python, script_path],
        capture_output=True, text=True, env=env,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"{os.path.basename(script_path)} failed (code {proc.returncode}):\n{proc.stderr}"
        )
    result = None
    for line in proc.stdout.splitlines():
        line = line.strip()
        if line.startswith(_SENTINEL):
            result = json.loads(line[len(_SENTINEL):].strip())
    if result is None:
        raise RuntimeError(
            f"{os.path.basename(script_path)} produced no result line:\n{proc.stdout}"
        )
    return result


def collect(script_paths, interpreters=None, env=None):
    import pandas as pd

    interpreters = interpreters or {}
    rows = []
    for path in script_paths:
        name = os.path.basename(path)
        python = interpreters.get(name)
        tag = f" [{python}]" if python else ""
        print(f"[bench] running {name}{tag} ...", flush=True)
        try:
            rows.append(run_backend(path, python=python, env=env))
        except Exception as exc:
            print(f"[bench] SKIP {name}: {exc}", file=sys.stderr, flush=True)
    return pd.DataFrame(rows)
