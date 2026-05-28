import os

import numpy as np
import torch


def artifacts_dir(case_dir: str) -> str:
    path = os.path.join(case_dir, "artifacts")
    os.makedirs(path, exist_ok=True)
    return path


def save_weights(model: torch.nn.Module, case_dir: str) -> str:
    path = os.path.join(artifacts_dir(case_dir), "weights.pt")
    torch.save(model.state_dict(), path)
    return path


def load_weights(model: torch.nn.Module, case_dir: str, device: str = "cpu") -> torch.nn.Module:
    path = os.path.join(artifacts_dir(case_dir), "weights.pt")
    state = torch.load(path, map_location=device)
    model.load_state_dict(state)
    return model


def save_inputs(case_dir: str, **arrays: np.ndarray) -> str:
    path = os.path.join(artifacts_dir(case_dir), "inputs.npz")
    np.savez(path, **arrays)
    return path


def load_inputs(case_dir: str) -> "np.lib.npyio.NpzFile":
    path = os.path.join(artifacts_dir(case_dir), "inputs.npz")
    return np.load(path)
