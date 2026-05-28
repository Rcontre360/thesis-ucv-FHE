import os
import tomllib
from typing import Any

from bench.paths import BENCH_DIR, SDK_ROOT

_CONFIG = BENCH_DIR / "envs.toml"


def _load_config() -> dict[str, Any]:
    if not _CONFIG.exists():
        return {}
    with open(_CONFIG, "rb") as f:
        return tomllib.load(f)


def _venv_python(venv: str) -> str:
    venv = venv if os.path.isabs(venv) else os.path.join(SDK_ROOT, venv)
    return os.path.join(venv, "bin", "python")


def interpreter_for(bid: str) -> str:
    cfg = _load_config()
    venv = cfg.get("interpreters", {}).get(bid, cfg["default"])
    return _venv_python(venv)
