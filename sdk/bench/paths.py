from pathlib import Path

BENCH_DIR: Path = Path(__file__).resolve().parent
SDK_ROOT: Path = BENCH_DIR.parent


def case_dir(case: str) -> str:
    return str(BENCH_DIR / case)
