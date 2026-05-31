import os
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

SEED: int = 30391915
N_CALIB: int = 200

BENCH_DIR: Path = Path(__file__).resolve().parent.parent
SDK_ROOT: Path = BENCH_DIR.parent
CONFIG_FILE: Path = BENCH_DIR / "config.toml"

# Env-var contract between the orchestrator and each subprocess.
ENV_VRAM_BASELINE: str = "BENCH_VRAM_BASELINE_BYTES"   # GPU floor (bytes) to subtract from vram_mb
ENV_RESULT_FILE: str   = "BENCH_RESULT_FILE"           # path where the child writes its result JSON
ENV_LATENCY_N: str     = "BENCH_LATENCY_N"             # override SampleCounts.latency for this run
ENV_ACCURACY_N: str    = "BENCH_ACCURACY_N"            # override SampleCounts.accuracy for this run
ENV_BENCH_CASE: str    = "BENCH_CASE"                  # case name, so samples_for can pick [samples.<case>]


def case_dir(case: str) -> str:
    return str(BENCH_DIR / case)


@dataclass
class SampleCounts:
    latency: int    # real-encrypted inferences to time
    accuracy: int   # samples for accuracy/fidelity (cheap clear-equivalent mode if the backend has one)


def _load() -> dict[str, Any]:
    if not CONFIG_FILE.exists():
        return {}
    with open(CONFIG_FILE, "rb") as f:
        return tomllib.load(f)


def _venv_python(venv: str) -> str:
    venv = venv if os.path.isabs(venv) else os.path.join(SDK_ROOT, venv)
    return os.path.join(venv, "bin", "python")


def interpreter_for(bid: str) -> str:
    cfg = _load().get("interpreters", {})
    return _venv_python(cfg.get(bid, cfg["default"]))


def samples_for(bid: str, case: str | None = None) -> SampleCounts:
    cfg = _load().get("samples", {})
    case = case or os.environ.get(ENV_BENCH_CASE)
    case_cfg = cfg.get(case, {}) if case else {}
    entry = case_cfg.get(bid) or cfg.get(bid) or cfg["default"]
    return SampleCounts(latency=int(entry["latency"]), accuracy=int(entry["accuracy"]))


def resolve_samples(bid: str) -> SampleCounts:
    counts = samples_for(bid)
    latency = int(os.environ.get(ENV_LATENCY_N) or counts.latency)
    accuracy = int(os.environ.get(ENV_ACCURACY_N) or counts.accuracy)
    return SampleCounts(latency=latency, accuracy=accuracy)
