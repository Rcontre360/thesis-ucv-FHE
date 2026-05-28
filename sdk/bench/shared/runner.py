import os
import json


def emit(result: dict) -> None:
    path = os.environ.get("BENCH_RESULT_FILE")
    if path:
        with open(path, "w") as f:
            json.dump(result, f)
    else:
        print(json.dumps(result))


def read_result(path: str) -> dict | None:
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)
