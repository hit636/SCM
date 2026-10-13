"""Bind exact-paper claims to externally supplied experimental protocol records."""

import json
from pathlib import Path
from fame_scm.reporting import content_hash


def protocol_snapshot(cfg, panel):
    return {
        "schema_version": 1,
        "data_sha256": panel.audit["sha256"],
        "cohort_hash": content_hash(panel.ids),
        "origin_lists": {k: [int(o) for o in v] for k, v in panel.origins.items()},
        "stage_intervals": {
            k: {"start": int(v.start), "end": int(v.end)}
            for k, v in panel.stages.items()
        },
        "experts_hash": content_hash(cfg["experts"]),
        "forecast_hash": content_hash(cfg["forecast"]),
        "competence_loss": cfg["scm"].get("competence_loss", "rmse"),
        "metric_scale": panel.audit.get("metric_scale", "raw"),
        "data_policy_hash": content_hash(
            {k: v for k, v in cfg["data"].items() if k != "path"}
        ),
        "selector_protocol_hash": content_hash(
            {
                k: cfg.get(k, {})
                for k in [
                    "features",
                    "scm",
                    "teacher",
                    "router",
                    "baselines",
                    "budgets",
                ]
            }
        ),
        "evaluation_protocol_hash": content_hash(cfg["evaluation"]),
        "training_seed": cfg["seed"],
    }


def validate_protocol(cfg, panel):
    actual = protocol_snapshot(cfg, panel)
    settings = cfg.get("reproduction", {})
    path = settings.get("author_protocol")
    strict = (
        settings.get("strict", False) or cfg.get("result_kind") == "paper_reproduction"
    )
    if strict and not path:
        raise ValueError(
            "Exact-paper reproduction requires reproduction.author_protocol; original cutoffs/cohort/origins/expert settings were not supplied"
        )
    if path:
        expected = json.loads(Path(path).read_text(encoding="utf-8"))
        missing = set(actual) - set(expected)
        if missing:
            raise ValueError(f"Author protocol fields missing: {sorted(missing)}")
        differences = [k for k, v in actual.items() if expected[k] != v]
        if differences:
            raise ValueError(f"Author protocol mismatch: {differences}")
    return {
        "status": "author_protocol_matched" if path else "documented_reimplementation",
        "strict": bool(strict),
        "snapshot": actual,
        "limitation": "A matching protocol binds inputs/settings; it does not attest that paper scores or hardware timings have been reproduced",
    }
