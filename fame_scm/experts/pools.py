"""Explicit, frozen engineering pool manifests (fixed expert settings)."""

from copy import deepcopy

BASE10 = [
    "sarima",
    "ets",
    "prophet",
    "tsb",
    "linear",
    "xgboost",
    "lightgbm",
    "dlinear",
    "timemixer",
    "timesnet",
]
EXTRA15 = ["catboost", "nbeats", "nhits", "patchtst", "deepar"]
EXTRA20 = ["tcn", "gru", "tft", "mamba", "chronos"]
REVISION = "129a2d204d6ab4205a0049a160402deefaf161b2"
ALTERNATIVES = {
    "sarima": {"order": [2, 0, 0]},
    "ets": {"damped_trend": True},
    "prophet": {"changepoint_prior_scale": 0.1},
    "tsb": {"alpha": 0.2},
    "linear": {"ridge_alpha": 10.0},
    "xgboost": {"tree_depth": 6},
    "lightgbm": {"num_leaves": 31},
    "dlinear": {"moving_avg": 13},
    "timemixer": {"d_model": 32, "d_ff": 64},
    "timesnet": {"d_model": 32, "d_ff": 64},
    "catboost": {"depth": 7},
    "nbeats": {"mlp_units": [[64, 64], [64, 64], [64, 64]]},
    "nhits": {"mlp_units": [[64, 64], [64, 64], [64, 64]]},
    "patchtst": {"hidden_size": 32, "patch_len": 8, "stride": 4},
    "deepar": {"lstm_hidden_size": 64},
    "tcn": {"encoder_hidden_size": 64},
    "gru": {"encoder_hidden_size": 64},
    "tft": {"hidden_size": 64},
    "mamba": {"width": 32, "state_size": 16},
    "chronos": {"num_samples": 40},
}


def configure_pool(cfg, size):
    if size not in [5, 10, 15, 20, 40]:
        raise ValueError("Pool must be 5/10/15/20/40")
    c = deepcopy(cfg)
    names = (
        ["sarima", "tsb", "lightgbm", "dlinear", "timemixer"]
        if size == 5
        else BASE10 + (EXTRA15 if size >= 15 else []) + (EXTRA20 if size >= 20 else [])
    )
    specs = {n: {"kind": n, "parameters": {}} for n in names}
    if size == 40:
        for n in list(names):
            specs[n + "__variant"] = {
                "kind": n,
                "parameters": deepcopy(ALTERNATIVES[n]),
            }
        names += [n + "__variant" for n in list(names)]
    c["experts"]["names"] = names
    c["experts"]["specs"] = specs
    c["experts"].setdefault("chronos_model", "amazon/chronos-t5-tiny")
    c["experts"].setdefault("chronos_revision", REVISION)
    c["cost_profiling"] = {**c.get("cost_profiling", {}), "enabled": True}
    c["experts"]["manifest_origin"] = "fixed repository expert configurations"
    return c
