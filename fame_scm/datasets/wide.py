"""Timestamp-preserving wide CSV adapter for Electricity and Traffic.

Each numeric column is one entity; all splits are chronological. Missing values
remain masked. Standardization fits only the training interval, per entity.
"""

from pathlib import Path
import csv
import hashlib
import json
import numpy as np
import pandas as pd
from fame_scm.datasets.snbc import SalesPanel
from fame_scm.datasets.data import (
    StageInterval,
    admissible_origins,
    validate_stage_intervals,
)
from fame_scm.datasets.temporal import time_step


def _stages(cfg, dates):
    split = cfg["split"]
    size = len(dates)
    if split.get("cutoffs"):
        if len(split["cutoffs"]) != 3:
            raise ValueError(
                "Provide exactly three full train/mining/calibration cutoff timestamps"
            )
        cutoffs = [pd.Timestamp(x) for x in split["cutoffs"]]
        if any(x not in dates for x in cutoffs):
            raise ValueError("Each cutoff must be an exact timestamp on the data grid")
        ends = [int(dates.get_loc(x)) for x in cutoffs]
    else:
        fractions = np.asarray(split["fractions"], float)
        if (
            len(fractions) != 4
            or np.any(fractions <= 0)
            or not np.isclose(fractions.sum(), 1)
        ):
            raise ValueError("Four positive split fractions must sum to one")
        ends = (np.floor(np.cumsum(fractions)[:3] * size).astype(int) - 1).tolist()
    bounds = [-1, *ends, size - 1]
    stages = {
        name: StageInterval(bounds[i] + 1, bounds[i + 1])
        for i, name in enumerate(["train", "mining", "calibration", "test"])
    }
    validate_stage_intervals(stages)
    f = cfg["forecast"]
    origins = {}
    if f["lookback"] < 1 or f["horizon"] < 1 or f["origin_stride"] < 1:
        raise ValueError("Lookback, horizon and origin_stride must be positive")
    if stages["train"].end + 1 < f["lookback"] + f["horizon"]:
        raise ValueError(
            "Training interval needs at least lookback + horizon time steps"
        )
    for name in ["mining", "calibration", "test"]:
        legal = admissible_origins(
            range(f["lookback"] - 1, size), stages[name], f["horizon"]
        )
        legal = legal[:: f["origin_stride"]]
        if f.get("max_origins_per_stage"):
            legal = legal[: f["max_origins_per_stage"]]
        if not len(legal):
            raise ValueError(
                f'{name} has no full {f["horizon"]}-step horizon; enlarge this interval or reduce horizon'
            )
        origins[name] = legal.tolist()
    return stages, origins


def load_wide_panel(cfg, *, inference=False, scaling=None):
    d = cfg["data"]
    f = cfg["forecast"]
    path = Path(d["path"])
    frequency = time_step(cfg)
    with path.open(encoding=d.get("encoding", "utf-8-sig"), newline="") as handle:
        header = next(csv.reader(handle), [])
    if not header or len(header) != len(set(header)):
        raise ValueError("CSV must have a nonempty, unique header")
    frame = pd.read_csv(path, encoding=d.get("encoding", "utf-8-sig"))
    date_column = d.get("date", "date")
    if date_column not in frame:
        raise ValueError(
            f"Missing timestamp column {date_column!r}; configure data.date"
        )
    dates = pd.DatetimeIndex(
        pd.to_datetime(frame.pop(date_column), format="mixed", errors="raise")
    )
    if dates.hasnans or dates.tz is not None:
        raise ValueError(
            "Use non-null, timezone-naive timestamps in one declared time zone"
        )
    if dates.has_duplicates:
        raise ValueError(
            "Duplicate timestamps; do not aggregate hourly samples into dates"
        )
    if len(dates) < 2:
        raise ValueError("At least two timestamps are required")
    frame.index = dates
    frame = frame.sort_index()
    columns = d.get("value_columns")
    if columns is None:
        columns = [c for c in frame.columns if c not in d.get("drop_columns", [])]
    if (
        not columns
        or len(columns) != len(set(columns))
        or any(c not in frame for c in columns)
    ):
        raise ValueError(
            "value_columns must select unique existing numeric series columns"
        )
    frame = frame[columns].apply(pd.to_numeric, errors="raise")
    dates = pd.date_range(frame.index[0], frame.index[-1], freq=frequency)
    if not frame.index.isin(dates).all():
        raise ValueError("Timestamps are not aligned to configured data.frequency")
    missing_steps = len(dates) - len(frame)
    if missing_steps and d.get("missing_timestamps", "error") != "reindex":
        raise ValueError(
            f"{missing_steps} missing time steps; set missing_timestamps: reindex to keep them masked"
        )
    frame = frame.reindex(dates)
    values = frame.to_numpy(dtype=float).T
    if np.isinf(values).any():
        raise ValueError(
            "Infinite values are invalid; missing values must be blank/NaN"
        )
    valid = np.isfinite(values)
    stages, origins = ({}, {}) if inference else _stages(cfg, dates)
    standardization = d.get("standardization", "none")
    if standardization not in ["none", "train_zscore"]:
        raise ValueError("standardization must be none or train_zscore")
    if standardization == "train_zscore" and cfg["forecast"].get(
        "clip_nonnegative", True
    ):
        raise ValueError("train_zscore requires forecast.clip_nonnegative: false")
    if "tsb" in cfg["experts"]["names"] and (
        standardization != "none" or (values[valid] < 0).any()
    ):
        raise ValueError(
            "TSB requires nonnegative counts; remove tsb from this expert pool"
        )
    ids = []
    arrays = []
    masks = []
    transforms = {}
    excluded = []
    for i, column in enumerate(columns):
        entity = json.dumps(
            [d.get("dataset", "timeseries"), column],
            ensure_ascii=False,
            separators=(",", ":"),
        )
        y = values[i]
        mask = valid[i]
        if inference:
            if (
                mask[-f["lookback"] :].sum()
                < f["lookback"] * f["min_valid_target_fraction"]
            ):
                raise ValueError(f"Insufficient recent observed history for {column}")
            if len(y) < f["lookback"]:
                raise ValueError("Insufficient history for prediction")
            if standardization == "train_zscore":
                if scaling is None or entity not in scaling:
                    raise ValueError(
                        f"No frozen training scaler for {column}; use fitted series IDs"
                    )
                transform = scaling[entity]
            else:
                transform = {"mean": 0.0, "scale": 1.0}
        else:
            train = y[: stages["train"].end + 1][mask[: stages["train"].end + 1]]
            if len(train) < d.get("min_observed_train_steps", f["lookback"]):
                excluded.append(column)
                continue
            transform = (
                {"mean": float(train.mean()), "scale": max(float(train.std()), 1e-8)}
                if standardization == "train_zscore"
                else {"mean": 0.0, "scale": 1.0}
            )
            if standardization == "train_zscore" and float(train.std()) < 1e-8:
                transform["scale"] = 1.0
            # Do not silently select a cohort using future target availability.
            for stage, times in origins.items():
                if any(
                    mask[o + 1 : o + f["horizon"] + 1].mean()
                    < f["min_valid_target_fraction"]
                    for o in times
                ):
                    raise ValueError(
                        f"{column}: insufficient observed targets in {stage}; inspect missingness and set an explicit coverage policy"
                    )
        arrays.append(np.where(mask, (y - transform["mean"]) / transform["scale"], 0.0))
        masks.append(mask)
        ids.append(entity)
        transforms[entity] = transform
        if d.get("max_series") and len(ids) >= d["max_series"]:
            break
    if not ids:
        raise ValueError("No series meet the training observation requirement")
    # Wide data has no sales metadata; shared immutable empty context avoids N*T copies.
    context = pd.DataFrame(index=dates)
    audit = {
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "format": "wide_csv",
        "dataset": d.get("dataset", "timeseries"),
        "frequency": str(frequency),
        "input_rows": len(frame) - missing_steps,
        "time_steps": len(dates),
        "entities": len(columns),
        "retained_entities": len(ids),
        "missing_timestamps": missing_steps,
        "missing_values": int((~valid).sum()),
        "excluded_training_columns": excluded,
        "cohort_selection": "CSV column order, training eligibility only, then max_series cap",
        "standardization": standardization,
        "metric_scale": (
            "training_zscore" if standardization == "train_zscore" else "raw"
        ),
        "scalers": transforms,
        "stage_dates": {
            s: [dates[v.start].isoformat(), dates[v.end].isoformat()]
            for s, v in stages.items()
        },
        "origins": {
            s: [dates[o].isoformat() for o in times] for s, times in origins.items()
        },
    }
    return SalesPanel(
        ids,
        dates,
        np.asarray(arrays),
        np.asarray(masks),
        [context] * len(ids),
        stages,
        origins,
        audit,
    )
