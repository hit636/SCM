"""SNBC daily data adapter. No sales amount or future metadata enters labels/features."""

from dataclasses import dataclass
from pathlib import Path
import hashlib, json
import numpy as np
import pandas as pd
from fame_scm.datasets.data import (
    StageInterval,
    admissible_origins,
    validate_stage_intervals,
)

CATEGORICAL = [
    "merc_brand_code",
    "merc_type_code",
    "machine_type",
    "scene_code",
    "city_name",
]
NUMERIC = ["merc_sale_price", "coupon_amount", "capacity"]


@dataclass
class SalesPanel:
    ids: list
    dates: pd.DatetimeIndex
    y: np.ndarray
    valid: np.ndarray
    frames: list
    stages: dict
    origins: dict
    audit: dict

    def history(self, i, origin, lookback, fill=0.0):
        x = self.y[i, max(0, origin - lookback + 1) : origin + 1].copy()
        mask = self.valid[i, max(0, origin - lookback + 1) : origin + 1]
        x = np.where(mask, x, fill)
        return np.pad(x, (max(0, lookback - len(x)), 0), constant_values=fill).astype(
            np.float32
        )

    def target(self, i, origin, horizon):
        return (
            self.y[i, origin + 1 : origin + horizon + 1].copy(),
            self.valid[i, origin + 1 : origin + horizon + 1].copy(),
        )


def read_sales(cfg):
    d = cfg["data"]
    path = Path(d["path"])
    frame = pd.read_csv(path, encoding=d["encoding"], dtype=str)
    input_rows = int(len(frame))
    duplicate_policy = d.get("duplicates", "first")
    if duplicate_policy not in ["first", "error", "sum"]:
        raise ValueError("duplicates must be first, error or sum")
    required = [d["date"], *d["entity_columns"], d["target"]]
    missing = [c for c in required if c not in frame]
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    if frame[[d["date"], *d["entity_columns"]]].isna().any().any():
        raise ValueError("Required date/IDs contain nulls")
    frame[d["target"]] = frame[d["target"]].replace(r"^\s*$", np.nan, regex=True)
    blank_targets = int(frame[d["target"]].isna().sum())
    if blank_targets:
        if d["missing_dates"] == "zero":
            frame[d["target"]] = frame[d["target"]].fillna("0")
        else:
            raise ValueError(
                "Required target contains nulls; use missing_dates: zero only when blanks mean zero sales"
            )
    frame["_date"] = pd.to_datetime(
        frame[d["date"]], errors="raise", format="mixed"
    ).dt.normalize()
    frame["_id"] = frame[d["entity_columns"]].apply(
        lambda x: json.dumps(x.tolist(), ensure_ascii=False, separators=(",", ":")),
        axis=1,
    )
    frame["_y"] = pd.to_numeric(frame[d["target"]], errors="raise")
    if not np.isfinite(frame["_y"]).all():
        raise ValueError("Nonfinite target")
    if (frame["_y"] < 0).any():
        if d["negative_target"] == "clip_zero":
            frame["_y"] = frame["_y"].clip(lower=0)
        else:
            raise ValueError("Negative quantity: choose an explicit returns policy")
    dup = int(frame.duplicated(["_id", "_date"]).sum())
    if dup:
        if duplicate_policy == "first":
            # Deduplicate before sorting: "first" means first in the source CSV.
            frame = frame.drop_duplicates(["_id", "_date"], keep="first").copy()
        elif duplicate_policy == "sum":
            agg = {c: "last" for c in frame if c not in ["_id", "_date", "_y"]}
            agg["_y"] = "sum"
            frame = frame.groupby(["_id", "_date"], as_index=False).agg(agg)
        else:
            raise ValueError(
                f"{dup} duplicate entity/date rows; configure duplicates: first to retain the first source row, or sum for additive records"
            )
    return frame.sort_values(["_id", "_date"]), {
        "input_rows": input_rows,
        "output_rows": int(len(frame)),
        "duplicate_rows": dup,
        "duplicates_policy": duplicate_policy,
        "dropped_duplicate_rows": dup if duplicate_policy == "first" else 0,
        "blank_target_rows_filled_zero": (
            blank_targets if d["missing_dates"] == "zero" else 0
        ),
        "entities": int(frame["_id"].nunique()),
        "start": str(frame["_date"].min().date()),
        "end": str(frame["_date"].max().date()),
        "columns": list(frame.columns),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def inspect_sales(cfg):
    frame, audit = read_sales(cfg)
    audit["calendar_days"] = (frame["_date"].max() - frame["_date"].min()).days + 1
    stats = frame.groupby("_id")["_date"].nunique().describe().to_dict()
    audit["observed_days_per_entity"] = {
        k: float(v) if np.isfinite(v) else None for k, v in stats.items()
    }
    audit["zero_quantity_rows"] = int((frame["_y"] == 0).sum())
    audit["note"] = (
        "capacity=0 is metadata, not a stockout mask; missing dates use configured policy"
    )
    return audit


def load_panel(cfg):
    frame, audit = read_sales(cfg)
    d = cfg["data"]
    f = cfg["forecast"]
    dates = pd.date_range(frame["_date"].min(), frame["_date"].max(), freq="D")
    t = len(dates)
    if cfg["split"]["cutoffs"]:
        ends = [
            int(dates.searchsorted(pd.Timestamp(x), side="right") - 1)
            for x in cfg["split"]["cutoffs"]
        ]
        if len(ends) != 3:
            raise ValueError("cutoffs must provide train/mining/calibration end dates")
    else:
        fractions = np.asarray(cfg["split"]["fractions"], float)
        if (
            len(fractions) != 4
            or not np.isclose(fractions.sum(), 1)
            or np.any(fractions <= 0)
        ):
            raise ValueError("Four positive split fractions must sum to 1")
        ends = (np.floor(np.cumsum(fractions)[:3] * t).astype(int) - 1).tolist()
    bounds = [-1, *ends, t - 1]
    stages = {
        s: StageInterval(bounds[j] + 1, bounds[j + 1])
        for j, s in enumerate(["train", "mining", "calibration", "test"])
    }
    validate_stage_intervals(stages)
    origins = {}
    for stage in ["mining", "calibration", "test"]:
        legal = admissible_origins(
            range(max(f["lookback"] - 1, stages[stage].start - 1), t),
            stages[stage],
            f["horizon"],
        )
        legal = legal[:: f["origin_stride"]]
        if f["max_origins_per_stage"]:
            legal = legal[: f["max_origins_per_stage"]]
        if not len(legal):
            raise ValueError(
                f'Insufficient data: {stage} needs a full {f["horizon"]}-day horizon after {f["lookback"]}-day history. CSV has {audit["input_rows"]} rows / {t} calendar days. Adjust split cutoffs or horizon based on the available history.'
            )
        origins[stage] = legal.tolist()
    ids = []
    ys = []
    masks = []
    frames = []
    excluded = []
    reason_counts = {
        "insufficient_training_observations": 0,
        "insufficient_valid_training_days": 0,
        "mining_target_coverage": 0,
        "calibration_target_coverage": 0,
        "test_target_coverage": 0,
    }
    training_counts = []
    valid_training_counts = []
    filled_zero_count = 0
    min_valid_train_days = d.get("min_valid_train_days", f["lookback"])
    for entity, g in frame.groupby("_id", sort=True):
        g = g.set_index("_date").reindex(dates)
        # Never fill before first seen date. Missing within the active history is
        # an explicit policy, not inferred from absence of sales records.
        seen = g["_y"].notna().to_numpy()
        y = g["_y"].fillna(0).to_numpy(float)
        valid = seen.copy()
        if d["missing_dates"] == "zero":
            valid[np.flatnonzero(seen)[0] :] = True
        availability = d.get("availability_column")
        if availability:
            if availability not in g:
                raise ValueError(f"Availability column absent: {availability}")
            valid &= (
                g[availability]
                .fillna("0")
                .astype(str)
                .isin(["1", "true", "True"])
                .to_numpy()
            )
        train_observed = int(seen[: stages["train"].end + 1].sum())
        train_valid = int(valid[: stages["train"].end + 1].sum())
        training_counts.append(train_observed)
        valid_training_counts.append(train_valid)
        filled_zero_count += int((valid & ~seen).sum())
        coverage = {
            stage: float(min(np.mean(valid[o + 1 : o + f["horizon"] + 1]) for o in os))
            for stage, os in origins.items()
        }
        reasons = []
        if train_observed < d["min_observed_train_days"]:
            reasons.append("insufficient_training_observations")
        if train_valid < min_valid_train_days:
            reasons.append("insufficient_valid_training_days")
        reasons.extend(
            f"{stage}_target_coverage"
            for stage, value in coverage.items()
            if value < f["min_valid_target_fraction"]
        )
        if reasons:
            for reason in reasons:
                reason_counts[reason] += 1
            excluded.append(
                {
                    "entity": entity,
                    "reasons": reasons,
                    "training_observed_days": train_observed,
                    "training_valid_days": train_valid,
                    "first_observed_date": str(dates[np.flatnonzero(seen)[0]].date()),
                    "last_observed_date": str(dates[np.flatnonzero(seen)[-1]].date()),
                    "min_target_coverage_by_stage": coverage,
                }
            )
            continue
        # Metadata is forward-filled only; changes remain origin-specific.
        for c in CATEGORICAL + NUMERIC:
            if c not in g:
                g[c] = np.nan
            g[c] = g[c].ffill()
        ids.append(entity)
        ys.append(y)
        masks.append(valid)
        frames.append(g)
        if d["max_series"] and len(ids) >= d["max_series"]:
            break
    audit.update(
        {
            "retained_entities": len(ids),
            "calendar_days": t,
            "missing_dates_policy": d["missing_dates"],
            "excluded_count_scanned": len(excluded),
            "excluded_examples": excluded[:10],
            "exclusion_reason_counts": reason_counts,
            "exclusion_counts_overlap": True,
            "scanned_entities": len(training_counts),
            "filled_zero_dates_scanned": filled_zero_count,
            "training_observed_days_scanned": {
                "min": int(min(training_counts)),
                "median": float(np.median(training_counts)),
                "max": int(max(training_counts)),
            },
            "training_valid_days_scanned": {
                "min": int(min(valid_training_counts)),
                "median": float(np.median(valid_training_counts)),
                "max": int(max(valid_training_counts)),
            },
            "eligibility_requirements": {
                "min_observed_train_days": d["min_observed_train_days"],
                "min_valid_train_days": min_valid_train_days,
                "min_valid_target_fraction": f["min_valid_target_fraction"],
                "min_valid_targets_per_origin": int(
                    np.ceil(f["horizon"] * f["min_valid_target_fraction"])
                ),
                "horizon": f["horizon"],
            },
            "stage_dates": {
                s: [str(dates[v.start].date()), str(dates[v.end].date())]
                for s, v in stages.items()
            },
            "origins": {
                s: [str(dates[o].date()) for o in os] for s, os in origins.items()
            },
            "cohort_selection": "lexicographic IDs after eligibility, capped by max_series",
        }
    )
    if not ids:
        from fame_scm.reporting import write_json

        report = Path(cfg.get("output", "runs/snbc")) / "data_audit.json"
        write_json(report, audit)
        raise ValueError(
            f"No eligible series among {len(training_counts)} scanned entities. "
            f'Training observations require >= {d["min_observed_train_days"]} days; '
            f"valid training days (including configured zero sales) require >= {min_valid_train_days}; "
            f'every evaluation origin requires >= {f["min_valid_target_fraction"]:.0%} '
            f'valid targets across {f["horizon"]} days. '
            f"Exclusion counts (overlap): {reason_counts}. "
            f'missing_dates={d["missing_dates"]}. Details saved to {report}.'
        )
    return SalesPanel(
        ids, dates, np.asarray(ys), np.asarray(masks), frames, stages, origins, audit
    )


def make_demo(path, *, series=32, days=420, seed=42):
    """Explicit synthetic SNBC-shaped data; no numerical paper targets used."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2024-01-01", periods=days)
    records = []
    for i in range(series):
        price = float(rng.choice([3.5, 5.0, 6.14, 8.0, 12.0]))
        base = rng.uniform(1, 10)
        for t, date in enumerate(dates):
            rate = max(
                0.1,
                base
                * (1 + 0.35 * np.sin(2 * np.pi * t / 7 + i))
                * (1 + (0.4 * t / days if i % 3 == 0 else 0)),
            )
            qty = int(rng.poisson(rate)) if (i % 4 or rng.random() < 0.25) else 0
            records.append(
                {
                    "date": date.strftime("%Y/%m/%d"),
                    "vem_id": f"DEMO_VM_{i//4:03d}",
                    "merc_id": f"DEMO_SKU_{i:04d}",
                    "daily_quantity": qty,
                    "merc_sale_price": price,
                    "coupon_amount": 0.0,
                    "daily_sales_amount": round(qty * price, 2),
                    "merc_brand_code": f"brand_{i%4}",
                    "merc_type_code": f"category_{i%3}",
                    "machine_type": i % 2,
                    "capacity": 0,
                    "scene_code": str(11000 + i % 3),
                    "city_name": ["Xi_an", "Jinan", "Qingdao"][i % 3],
                }
            )
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_csv(path, index=False, encoding="utf-8-sig")
    return {
        "result_kind": "synthetic_demo",
        "series": series,
        "days": days,
        "rows": len(records),
        "seed": seed,
    }
