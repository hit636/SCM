"""Default configuration, recursive YAML composition, and validation."""

from copy import deepcopy
from pathlib import Path
import re
import yaml

DEFAULTS = {
    "seed": 42,
    "device": "cpu",
    "threads": 1,
    "result_kind": "independent_reimplementation",
    "data": {
        "format": "snbc",
        "path": "data/daily_sales_data.csv",
        "encoding": "utf-8-sig",
        "date": "date",
        "entity_columns": ["vem_id", "merc_id"],
        "target": "daily_quantity",
        "missing_dates": "zero",
        "duplicates": "first",
        "negative_target": "error",
        "availability_column": None,
        "max_series": 128,
        "min_observed_train_days": 1,
        "min_valid_train_days": 49,
    },
    "split": {"fractions": [0.7, 0.05, 0.05, 0.2], "cutoffs": None},
    "forecast": {
        "lookback": 49,
        "horizon": 14,
        "seasonality": 7,
        "origin_stride": 14,
        "max_origins_per_stage": 2,
        "train_stride": 7,
        "min_valid_target_fraction": 0.8,
    },
    "experts": {
        "names": ["sarima", "tsb", "lightgbm", "dlinear", "timemixer"],
        "anchor": "lightgbm",
        "tree_estimators": 80,
        "neural_epochs": 8,
        "batch_size": 64,
        "learning_rate": 0.001,
        "max_train_windows": 4096,
        "sarima_maxiter": 30,
        "d_model": 16,
        "d_ff": 32,
        "e_layers": 1,
        "cost_weights": {
            "sarima": 0.8,
            "ets": 0.8,
            "prophet": 1.5,
            "tsb": 0.3,
            "linear": 0.6,
            "xgboost": 1.1,
            "lightgbm": 1.0,
            "dlinear": 2.0,
            "timemixer": 3.6,
            "timesnet": 4.1,
        },
    },
    "scm": {
        "rho": 0.4,
        "members": 5,
        "batches": 4,
        "epochs": 40,
        "batch_size": 64,
        "patience": 10,
        "learning_rate": 0.001,
        "rank_margin": 0.05,
        "lambda_rank": 0.2,
        "lambda_factor": 0.0001,
        "lambda_encoder": 0.0001,
        "ensemble_keep_probability": 0.85,
        "sd_ddof": 1,
        "mad_multiplier": 1.0,
        "scale_floor": 0.1,
        "alpha": 1.0,
        "beta": 0.5,
        "zeta": 0.1,
        "competence_loss": "rmse",
        "reconstruction": "huber",
        "huber_delta": 1.0,
        "row_normalization": True,
        "rank_margin_mode": "fixed",
    },
    "teacher": {"tau": 0.3, "eta": 0.5, "nu": 0.05},
    "router": {
        "epochs": 60,
        "patience": 15,
        "batch_size": 64,
        "learning_rate": 0.001,
        "dropout": 0.1,
        "prediction_loss": "mse",
        "lambda_pred": 1.0,
        "lambda_teacher": 1.0,
        "lambda_balance": 0.01,
        "lambda_cost": 0.01,
        "r": 2,
        "threshold": 0.05,
        "calibration_thresholds": [0.0, 0.05, 0.1],
        "consumer": "distilled",
        "operator": "soft_kl",
        "operator_temperature": 1.0,
    },
    "baselines": {
        "methods": [
            "lightgbm",
            "full_observation",
            "random_partial",
            "softimpute",
            "imc",
            "independent_lgbm",
            "scm_passive",
            "fame_scm_active",
        ],
        "epochs": 40,
        "neighbors": 5,
        "rank": 4,
        "softimpute_lambda": 0.1,
        "fformpp_steps": 20,
        "fformpp_features": 4,
        "fformpp_samples": 16,
        "pmf_latent_dim": 3,
    },
    "evaluation": {"hidden_audit": True, "mask_seeds": [0], "neural_seeds": [0]},
    "output": "runs/snbc",
}


def merge(base, override):
    out = deepcopy(base)
    for k, v in override.items():
        out[k] = (
            merge(out[k], v)
            if isinstance(v, dict) and isinstance(out.get(k), dict)
            else deepcopy(v)
        )
    return out


def load_settings(path=None):
    custom = {} if path is None else _read_config(Path(path))
    cfg = merge(DEFAULTS, custom)
    if not re.fullmatch(r"cpu|cuda(?::[0-9]+)?", str(cfg["device"])):
        raise ValueError("device must be cpu, cuda or cuda:<visible GPU index>")
    if not 0 < cfg["scm"]["rho"] <= 1:
        raise ValueError("rho must be in (0,1]")
    if cfg["data"]["missing_dates"] not in ["unknown", "zero"]:
        raise ValueError("missing_dates must be unknown or zero")
    if cfg["experts"]["anchor"] not in cfg["experts"]["names"]:
        raise ValueError("Anchor must belong to expert pool")
    if len(set(cfg["experts"]["names"])) != len(cfg["experts"]["names"]):
        raise ValueError("Duplicate expert IDs")
    for section in ["scm", "router"]:
        for key in ["epochs", "batch_size", "patience"]:
            if cfg[section][key] < 1:
                raise ValueError(f"{section}.{key} must be positive")
    if cfg["scm"]["members"] < 1 or cfg["scm"]["batches"] < 1:
        raise ValueError("SCM members and batches must be positive")
    if not 0 < cfg["scm"]["ensemble_keep_probability"] <= 1:
        raise ValueError("Ensemble keep probability must be in (0,1]")
    if cfg["router"]["r"] < 1 or not 0 <= cfg["router"]["threshold"] <= 1:
        raise ValueError("Invalid routing budget or threshold")
    if not cfg["evaluation"]["mask_seeds"] or not cfg["evaluation"]["neural_seeds"]:
        raise ValueError("Evaluation seed lists must be nonempty")
    if cfg["scm"]["competence_loss"] not in ["rmse", "mse", "mae", "wape", "smape"]:
        raise ValueError("Unknown competence loss metric")
    if cfg["scm"]["rank_margin_mode"] not in ["fixed", "acquired_rolling"]:
        raise ValueError("Unknown ranking margin mode")
    if cfg["scm"]["reconstruction"] not in ["huber", "mse"]:
        raise ValueError("Unknown reconstruction objective")
    if cfg["router"]["consumer"] not in ["distilled", "direct_teacher"]:
        raise ValueError("Unknown teacher consumer")
    if cfg["router"]["operator"] not in [
        "soft_kl",
        "sparsemax",
        "straight_through",
        "gumbel_top_r",
    ]:
        raise ValueError("Unknown router operator")
    if cfg["teacher"]["tau"] <= 0 or cfg["scm"]["huber_delta"] <= 0:
        raise ValueError("Temperature and Huber threshold must be positive")
    return cfg


def _read_config(path, stack=()):
    path = path.resolve()
    if path in stack:
        raise ValueError("Circular config extends: " + str(path))
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError("Config must be a mapping")
    parents = data.pop("extends", [])
    if isinstance(parents, str):
        parents = [parents]
    if not isinstance(parents, list):
        raise ValueError("extends must be a path or ordered path list")
    result = {}
    for parent in parents:
        result = merge(result, _read_config(path.parent / parent, stack + (path,)))
    return merge(result, data)
