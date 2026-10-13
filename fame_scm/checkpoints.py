"""Atomic trusted-local research checkpoints. Never unpickle untrusted input."""

import os, pickle
from pathlib import Path


def save_state(path, state):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".partial")
    with temp.open("wb") as f:
        pickle.dump(state, f, protocol=pickle.HIGHEST_PROTOCOL)
        f.flush()
        os.fsync(f.fileno())
    os.replace(temp, path)


def load_state(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def state_training_seed(state):
    """Use the saved mask/neural replicate, never silently collapse repeats."""
    cfg = state["cfg"]
    if "evidence_mask_id" not in state or "neural_seed" not in state:
        if any(len(cfg["evaluation"][k]) > 1 for k in ["mask_seeds", "neural_seeds"]):
            raise ValueError(
                "Legacy repeated checkpoint lacks mask/neural IDs; regenerate build state with version 2.0"
            )
    return (
        cfg["seed"]
        + 10000 * state.get("evidence_mask_id", 0)
        + state.get("neural_seed", 0)
    )
