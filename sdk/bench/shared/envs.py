import os
import tomllib

_HERE = os.path.dirname(os.path.abspath(__file__))
_BENCH_ROOT = os.path.dirname(_HERE)
_SDK_ROOT = os.path.dirname(_BENCH_ROOT)
_CONFIG = os.path.join(_BENCH_ROOT, "envs.toml")


def backend_id(script_name):
    name = os.path.basename(script_name)
    if name.startswith("run_") and name.endswith(".py"):
        return name[len("run_"):-len(".py")]
    return name


def _load_config():
    if not os.path.exists(_CONFIG):
        return {}
    with open(_CONFIG, "rb") as f:
        return tomllib.load(f)


def _venv_python(venv):
    venv = venv if os.path.isabs(venv) else os.path.join(_SDK_ROOT, venv)
    return os.path.join(venv, "bin", "python")


def interpreter_for(bid):
    """Resolve the python interpreter for a library id.

    Returns an absolute python path, or None to use the driver's own
    interpreter. Raises FileNotFoundError if an explicitly configured
    interpreter (env var or [interpreters] entry) does not exist.
    """
    cfg = _load_config()
    interpreters = cfg.get("interpreters", {})

    explicit = os.environ.get(f"BENCH_PYTHON_{bid.upper()}")
    if explicit:
        source, venv = "env var", explicit
    elif bid in interpreters:
        source, venv = "envs.toml", interpreters[bid]
    elif cfg.get("default"):
        source, venv = "default", cfg["default"]
    else:
        return None

    python = _venv_python(venv)
    if not os.path.exists(python):
        if source == "default":
            return None
        raise FileNotFoundError(
            f"interpreter for '{bid}' ({source}) not found: {python}. "
            f"Create it with ./scripts/setup_env.sh {bid}, or fix bench/envs.toml."
        )
    return python
