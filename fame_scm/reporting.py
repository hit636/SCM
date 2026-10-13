import hashlib
import json
from pathlib import Path
import numpy as np

COMPARISON_HASHES = (
    "cohort_hash",
    "split_hash",
    "origin_hash",
    "target_mask_hash",
    "feature_hash",
    "expert_pool_hash",
    "availability_hash",
    "metric_protocol_hash",
    "gate_version",
)


def content_hash(value):
    data = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(data.encode()).hexdigest()


def check_comparable(left, right):
    for key in COMPARISON_HASHES:
        if key not in left or key not in right or left[key] != right[key]:
            raise ValueError(f"Comparison protocol mismatch/missing: {key}")


def check_pooled_metric_identity(metrics, *, atol=1e-10):
    if metrics["rmse"] + atol < metrics["mae"]:
        raise ValueError("RMSE below MAE on common target weights")
    if metrics["wape"] is not None and metrics["wape"] > 0:
        implied = metrics["mae"] / metrics["wape"]
        if not np.isclose(implied, metrics["target_abs_mean"], atol=atol, rtol=1e-8):
            raise ValueError("MAE/WAPE target-denominator inconsistency")


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    import os

    temp = path.with_name(path.name + ".partial")
    with temp.open("w", encoding="utf-8") as f:
        f.write(
            json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
        )
        f.flush()
        os.fsync(f.fileno())
    os.replace(temp, path)
