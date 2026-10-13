"""Preflight executable dependencies and data/protocol before expensive runs."""

import importlib
import importlib.metadata
from pathlib import Path


def check_environment(cfg, check_data=False):
    modules = {
        "numpy",
        "pandas",
        "scipy",
        "sklearn",
        "yaml",
        "torch",
        "lightgbm",
        "statsmodels",
        "einops",
        "psutil",
    }
    kinds = {
        cfg["experts"].get("specs", {}).get(n, {}).get("kind", n)
        for n in cfg["experts"]["names"]
    }
    optional = {
        "xgboost": "xgboost",
        "prophet": "prophet",
        "catboost": "catboost",
        "chronos": "chronos",
    }
    modules.update(v for k, v in optional.items() if k in kinds)
    if kinds & {"nbeats", "nhits", "patchtst", "deepar", "tcn", "gru", "tft"}:
        modules.add("neuralforecast")
    if "fformpp_b" in cfg["baselines"]["methods"]:
        modules.update(["jax", "movingknots"])
    dependencies, import_errors = {}, {}
    for name in sorted(modules):
        try:
            importlib.import_module(name)
            dependencies[name] = True
        except Exception as exc:
            dependencies[name] = False
            import_errors[name] = f"{type(exc).__name__}: {exc}"
    missing = [name for name, present in dependencies.items() if not present]
    try:
        from fame_scm.devices import device_report

        device = device_report(cfg["device"], probe=True)
        device_error = None
    except Exception as exc:
        device = {"requested": cfg["device"], "probe": "failed"}
        device_error = f"{type(exc).__name__}: {exc}"
    result = {
        "dependencies": dependencies,
        "import_errors": import_errors,
        "missing": missing,
        "status": "ready" if not missing else "missing_dependencies",
        "device": device,
        "device_error": device_error,
        "data_path": cfg["data"]["path"],
        "data_exists": Path(cfg["data"]["path"]).exists(),
        "result_kind": cfg["result_kind"],
    }
    if not missing and device_error:
        result["status"] = "device_unavailable"
    if check_data:
        from fame_scm.datasets.loader import load_panel
        from fame_scm.reproduction import validate_protocol

        panel = load_panel(cfg)
        result["protocol"] = validate_protocol(cfg, panel)
        result["n"] = len(panel.ids)
        result["origins_by_stage"] = {k: len(v) for k, v in panel.origins.items()}
    return result
