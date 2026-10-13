"""Training-only execution profiling; outputs are never competence labels."""

import time, json
from pathlib import Path
from fame_scm.reporting import content_hash, write_json
import numpy as np
import torch


def profile_costs(pool, panel, cfg):
    spec = cfg.get("cost_profiling", {})
    names = pool.names
    if not spec.get("enabled", False):
        declared = cfg["experts"]["cost_weights"]
        return {
            "kind": "declared_weights_not_measured",
            "online": {k: float(declared[k]) for k in names},
            "offline": {k: float(declared[k]) for k in names},
            "events": [],
            "seconds": 0.0,
        }
    signature = content_hash(
        {
            "input": panel.audit["sha256"],
            "ids": panel.ids,
            "experts": cfg["experts"],
            "seed": cfg["seed"],
            "lookback": pool.lookback,
            "horizon": pool.h,
            "mining_origins": panel.origins["mining"],
            "device": str(pool.device),
            "torch_build": torch.__version__,
            "gpu_name": (
                torch.cuda.get_device_name(pool.device)
                if pool.device.type == "cuda"
                else None
            ),
        }
    )
    frozen = Path(spec["profile_path"]) if spec.get("profile_path") else None
    if frozen is not None and frozen.exists():
        result = json.loads(frozen.read_text())
        if result.get("signature") != signature:
            raise ValueError(
                "Frozen cost profile does not match data/cohort/expert/device protocol"
            )
        result.update(seconds=0.0, events=[], reused_profile=str(frozen))
        return result
    start = time.perf_counter()
    rng = np.random.default_rng(cfg["seed"])
    events = []
    cost = {}
    origin = panel.stages["train"].end - cfg["forecast"]["horizon"]
    if origin < pool.lookback - 1:
        raise ValueError("Not enough training history for cost profiling")
    avail = pool.availability([origin])
    for j, name in enumerate(names):
        rows = np.flatnonzero(avail[:, j])
        rows = rng.permutation(rows)[: spec.get("series", 16)]
        if not len(rows):
            raise ValueError(f"No training-only cost-profile rows for {name}")
        times = []
        for i in rows:
            for r in range(spec.get("repeats", 3) + 1):
                if pool.device.type == "cuda":
                    torch.cuda.synchronize(pool.device)
                t = time.perf_counter()
                pool.predict(name, int(i), origin)
                if pool.device.type == "cuda":
                    torch.cuda.synchronize(pool.device)
                elapsed = time.perf_counter() - t
                events.append(
                    {
                        "stage": "preparation_profile",
                        "expert": name,
                        "entity": panel.ids[int(i)],
                        "origin": origin,
                        "warmup": r == 0,
                        "seconds": elapsed,
                    }
                )
                if r:
                    times.append(elapsed)
        cost[name] = max(float(np.median(times)), 1e-9)
    anchor = cost[cfg["experts"]["anchor"]]
    result = {
        "signature": signature,
        "kind": "measured_training_only_scalar_execution_batch_size_1",
        "online": {k: v / anchor for k, v in cost.items()},
        "offline": {k: v * len(panel.origins["mining"]) for k, v in cost.items()},
        "median_seconds_per_request": cost,
        "events": events,
        "seconds": time.perf_counter() - start,
        "note": "Offline weights estimate one multi-origin cell; actual stage durations are separately measured. Statistical fitting is included. No claim of GPU batched-throughput profiling.",
    }

    if frozen is not None:
        write_json(frozen, result)
    return result
