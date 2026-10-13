"""Synthetic original-ten-pool -> five separately inserted real backends."""

from pathlib import Path
import sys, argparse

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fame_scm.settings import load_settings
from fame_scm.experts.pools import configure_pool
from fame_scm.datasets.snbc import make_demo
from fame_scm.orchestration.experiment import run_experiment
from fame_scm.orchestration.screening import run_insertion
from fame_scm.reporting import write_json
import torch


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    a = p.parse_args()
    root = Path(a.out)
    if root.exists() and any(root.iterdir()):
        raise ValueError("Fresh output required")
    root.mkdir(parents=True)
    torch.set_num_threads(1)
    make_demo(root / "demo.csv", series=12, days=420, seed=42)
    cfg = configure_pool(load_settings(), 10)
    cfg["data"].update(path=str((root / "demo.csv").resolve()), max_series=12)
    cfg["result_kind"] = "synthetic_insertion_validation"
    cfg["forecast"]["max_origins_per_stage"] = 1
    cfg["experts"].update(
        neural_epochs=1,
        tree_estimators=5,
        max_train_windows=12,
        nf_steps=1,
        sarima_maxiter=3,
    )
    cfg["scm"].update(epochs=2, members=2, batches=2)
    cfg["router"].update(epochs=2, calibration_thresholds=[0, 0.05])
    cfg["evaluation"]["hidden_audit"] = False
    cfg["research"] = {"save_state": True}
    cfg["cost_profiling"] = {"enabled": True, "series": 1, "repeats": 1}
    cfg["output"] = str(root / "base")
    run_experiment(cfg, methods=["fame_scm_active"])
    checkpoint = root / "base/fame_scm_active/mask_0/seed_0/build_state.pkl"
    rows = []
    for expert in ["catboost", "patchtst", "nhits", "deepar", "chronos"]:
        print("INSERTION", expert, flush=True)
        result = run_insertion(checkpoint, expert, root / expert)
        assert len(result) == 10
        rows.append(
            {
                "expert": expert,
                "runs": len(result),
                "all_frozen": all(
                    s["old_logits_and_parameters_frozen"] for s in result
                ),
            }
        )
    write_json(
        root / "validation_summary.json",
        {
            "kind": "synthetic_only",
            "old_pool_size": 10,
            "new_backends": rows,
            "completed_runs": 50,
        },
    )


if __name__ == "__main__":
    main()
