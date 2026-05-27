import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shared.runner import run_step, collect
from shared.envs import backend_id, interpreter_for

CASE_DIR = os.path.dirname(os.path.abspath(__file__))
CASE = os.path.basename(CASE_DIR)

BACKENDS = ["run_pytorch.py", "run_sdk.py", "run_cml.py", "run_orion.py"]


def main():
    run_step(os.path.join(CASE_DIR, "train.py"))

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

    df = collect(scripts, interpreters=interpreters)
    df.insert(0, "case", CASE)
    out = os.path.join(CASE_DIR, "results.csv")
    df.to_csv(out, index=False)
    print()
    print(df.to_string(index=False))
    print("saved", out)


if __name__ == "__main__":
    main()
