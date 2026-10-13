"""Memory-bounded M5/Favorita adapters. No leaderboard/test labels are consumed.

The stored format is numeric .npy + JSON (no pickle), with streaming source
hashes and train-only cohort selection. All date/return/zero conventions are
explicit in the manifest; absence of a sales record is not a stockout label.
"""

from pathlib import Path
import hashlib, json
import numpy as np
import pandas as pd
from fame_scm.datasets.snbc import SalesPanel
from fame_scm.datasets.data import (
    StageInterval,
    admissible_origins,
    validate_stage_intervals,
)
from fame_scm.reporting import write_json, content_hash


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def make_stages(dates, split, forecast):
    t = len(dates)
    cut = split.get("cutoffs")
    if cut:
        if len(cut) != 3:
            raise ValueError("Exactly three chronological cutoffs required")
        ends = [int(dates.searchsorted(pd.Timestamp(x), side="right") - 1) for x in cut]
    else:
        f = np.asarray(split["fractions"])
        if len(f) != 4 or np.any(f <= 0) or not np.isclose(f.sum(), 1):
            raise ValueError("Invalid stage fractions")
        ends = (np.floor(np.cumsum(f)[:3] * t).astype(int) - 1).tolist()
    b = [-1, *ends, t - 1]
    if any(b[i] >= b[i + 1] for i in range(4)):
        raise ValueError("Empty/overlapping stages")
    stages = {
        s: StageInterval(b[i] + 1, b[i + 1])
        for i, s in enumerate(["train", "mining", "calibration", "test"])
    }
    validate_stage_intervals(stages)
    origins = {}
    for s in ["mining", "calibration", "test"]:
        legal = admissible_origins(
            range(max(forecast["lookback"] - 1, stages[s].start - 1), t),
            stages[s],
            forecast["horizon"],
        )
        legal = legal[:: forecast["origin_stride"]]
        if forecast.get("max_origins_per_stage"):
            legal = legal[: forecast["max_origins_per_stage"]]
        if not len(legal):
            raise ValueError(f"{s} has no full-horizon origin")
        origins[s] = legal.tolist()
    return stages, origins


class ContextFrame:
    """A lightweight read-only frame: static categories do not repeat N x T strings."""

    def __init__(self, static, numeric, shared, t):
        self.static = static
        self.numeric = numeric
        self.shared = shared
        self.t = t

    @property
    def iloc(self):
        return self

    def __getitem__(self, key):
        if isinstance(key, (int, np.integer)):
            return {
                **self.static,
                **{k: v[key] for k, v in self.shared.items()},
                **{k: v[key] for k, v in self.numeric.items()},
            }
        if key in self.static:
            return pd.Series([self.static[key]])
        return pd.Series(self.numeric[key] if key in self.numeric else self.shared[key])


def _fresh(out):
    out = Path(out)
    if out.exists() and any(out.iterdir()):
        raise ValueError(f"Nonempty prepared output: {out}")
    out.mkdir(parents=True, exist_ok=True)
    return out


def _rank(ids, seed):
    return sorted(
        ids, key=lambda x: hashlib.sha256((str(seed) + "|" + x).encode()).hexdigest()
    )


def _save(out, ids, dates, y, valid, static, numeric, shared, manifest):
    if not len(ids):
        raise ValueError("Empty training-eligible cohort")
    np.save(out / "y.npy", np.asarray(y, np.float32))
    np.save(out / "valid.npy", np.asarray(valid, bool))
    for k, a in numeric.items():
        np.save(out / f"context_{k}.npy", np.asarray(a, np.float32))
    for k, a in shared.items():
        np.save(out / f"calendar_{k}.npy", np.asarray(a, np.float32))
    manifest.update(
        {
            "schema_version": 2,
            "ids": ids,
            "dates": [str(x.date()) for x in dates],
            "static": static,
            "numeric_columns": list(numeric),
            "shared_columns": list(shared),
            "categorical_columns": list(static[0]),
            "cohort_hash": content_hash(ids),
            "n": len(ids),
            "cohort_policy": "training eligibility then seeded SHA256 order; never validation/test sales statistics",
            "sha256": content_hash(manifest["source_hashes"]),
        }
    )
    write_json(out / "manifest.json", manifest)
    write_json(
        out / "integrity.json",
        {p.name: sha256(p) for p in sorted(out.iterdir()) if p.is_file()},
    )
    return manifest


def prepare_m5(
    raw,
    out,
    *,
    train_end,
    max_series=30490,
    seed=42,
    min_train_days=15,
    sales_file="sales_train_evaluation.csv",
):
    raw = Path(raw)
    out = _fresh(out)
    end = pd.Timestamp(train_end)
    sources = [raw / sales_file, raw / "calendar.csv", raw / "sell_prices.csv"]
    cal = pd.read_csv(sources[1])
    cal["date"] = pd.to_datetime(cal["date"])
    wide = pd.read_csv(sources[0])
    days = sorted([c for c in wide if c.startswith("d_")], key=lambda s: int(s[2:]))
    if not days:
        raise ValueError("M5 wide daily labels absent")
    cal = cal.set_index("d").loc[days]
    dates = pd.DatetimeIndex(cal.date)
    if not dates.equals(pd.date_range(dates[0], dates[-1])):
        raise ValueError("Non-daily M5 calendar")
    if not dates[0] <= end < dates[-1]:
        raise ValueError("Training cutoff outside observed sales dates")
    train = dates <= end
    prices = pd.read_csv(sources[2])
    keys = ["store_id", "item_id", "wm_yr_wk"]
    if prices.duplicated(keys).any():
        raise ValueError("Duplicate weekly M5 prices")
    if prices.sell_price.isna().any() or (prices.sell_price <= 0).any():
        raise ValueError("Invalid M5 prices")
    groups = {
        k: g.set_index("wm_yr_wk").sell_price
        for k, g in prices.groupby(["store_id", "item_id"], sort=False)
    }
    candidates = {}
    eligibility = {}
    for idx, row in wide.iterrows():
        key = (row.store_id, row.item_id)
        price = groups.get(key)
        if price is None:
            continue
        p = price.reindex(cal.wm_yr_wk).to_numpy(float)
        listed = np.isfinite(p)
        if not listed.any():
            continue
        valid = np.arange(len(dates)) >= np.flatnonzero(listed)[0]
        # Missing later prices are unknown metadata, not evidence of a stockout.
        if int((valid & train).sum()) < min_train_days:
            continue
        entity = json.dumps(
            [str(row.store_id), str(row.item_id)], separators=(",", ":")
        )
        if entity in candidates:
            raise ValueError("Duplicate M5 store-item ID")
        candidates[entity] = (idx, valid, p)
        eligibility[entity] = int((valid & train).sum())
    ids = _rank(candidates, seed)
    ids = ids[:max_series] if max_series else ids
    ys = []
    masks = []
    ps = []
    snap = []
    static = []
    for entity in ids:
        idx, valid, p = candidates[entity]
        row = wide.loc[idx]
        y = row[days].to_numpy(np.float32)
        if not np.isfinite(y).all() or np.any(y < 0):
            raise ValueError("Invalid M5 sales")
        ys.append(y)
        masks.append(valid)
        ps.append(pd.Series(p).ffill().to_numpy())
        snap.append(cal["snap_" + str(row.state_id)].to_numpy(float))
        static.append(
            {
                k: str(row[k])
                for k in ["store_id", "item_id", "dept_id", "cat_id", "state_id"]
            }
        )
    shared = {
        "event": (cal.event_name_1.notna() | cal.event_name_2.notna()).to_numpy(float)
    }
    return _save(
        out,
        ids,
        dates,
        ys,
        masks,
        static,
        {"sell_price": ps, "snap": snap},
        shared,
        {
            "dataset": "m5",
            "train_end": str(end.date()),
            "seed": seed,
            "eligible_training_series": len(candidates),
            "source_hashes": {p.name: sha256(p) for p in sources},
            "sales_file": sales_file,
            "zero_policy": "pre-first-listed-price masked; later reported zeros retained; no inferred stockouts",
            "context_policy": "origin-observed price/SNAP/event only; ffill price only; no future price inputs",
            "min_train_days": min_train_days,
        },
    )


def prepare_favorita(
    raw,
    out,
    *,
    train_end,
    start="2013-01-01",
    end="2017-08-15",
    max_series=18000,
    seed=42,
    min_train_days=15,
    chunksize=500000,
    returns="clip_zero",
):
    raw = Path(raw)
    out = _fresh(out)
    start = pd.Timestamp(start)
    end = pd.Timestamp(end)
    cut = pd.Timestamp(train_end)
    if not start <= cut < end:
        raise ValueError("Invalid Favorita date range")
    if returns not in ["clip_zero", "error"]:
        raise ValueError("Returns policy must be clip_zero or error")
    sources = [
        raw / "train.csv",
        raw / "items.csv",
        raw / "stores.csv",
        raw / "oil.csv",
        raw / "holidays_events.csv",
    ]
    use = ["date", "store_nbr", "item_nbr", "unit_sales", "onpromotion"]
    counts = {}
    first = {}
    neg = 0
    scanned = 0

    def chunks():
        for c in pd.read_csv(sources[0], usecols=use, chunksize=chunksize):
            c["date"] = pd.to_datetime(c["date"], errors="raise")
            c = c[(c.date >= start) & (c.date <= end)].copy()
            if not np.isfinite(c.unit_sales).all():
                raise ValueError("Nonfinite Favorita sales")
            if c[["store_nbr", "item_nbr"]].isna().any().any():
                raise ValueError("Missing Favorita identity")
            yield c

    # Pass 1 only looks at TRAIN records for eligibility and identities.
    for c in chunks():
        scanned += len(c)
        neg += int((c.unit_sales < 0).sum())
        if returns == "error" and (c.unit_sales < 0).any():
            raise ValueError("Returns found; choose explicit clip_zero policy")
        tr = c[c.date <= cut]
        for key, g in tr.groupby(["store_nbr", "item_nbr"], sort=False):
            key = tuple(map(int, key))
            counts[key] = counts.get(key, 0) + len(g)
            first[key] = min(first.get(key, g.date.min()), g.date.min())
    key_by_id = {
        json.dumps([str(s), str(i)], separators=(",", ":")): (s, i)
        for (s, i), v in counts.items()
        if v >= min_train_days
    }
    ids = _rank(key_by_id, seed)
    ids = ids[:max_series] if max_series else ids
    if not ids:
        raise ValueError("No Favorita training-eligible series")
    keys = [key_by_id[x] for x in ids]
    lookup = {k: i for i, k in enumerate(keys)}
    dates = pd.date_range(start, end)
    n = len(ids)
    t = len(dates)
    # Arrays are disk-backed while scanning the 100M+ row transaction file.
    y = np.lib.format.open_memmap(
        out / "_sales_work.npy", mode="w+", dtype="float32", shape=(n, t)
    )
    y[:] = 0
    promo = np.lib.format.open_memmap(
        out / "_promo_work.npy", mode="w+", dtype="float32", shape=(n, t)
    )
    promo[:] = np.nan
    seen = np.zeros((n, t), bool)
    for c in chunks():
        row = np.array(
            [lookup.get((int(s), int(i)), -1) for s, i in zip(c.store_nbr, c.item_nbr)]
        )
        take = row >= 0
        c = c.loc[take]
        row = row[take]
        col = (c.date - start).dt.days.to_numpy()
        flat = row * t + col
        if len(np.unique(flat)) != len(flat) or seen[row, col].any():
            raise ValueError("Duplicate Favorita date/store/item rows")
        seen[row, col] = True
        y[row, col] = c.unit_sales.clip(lower=0).to_numpy(np.float32)
        v = c.onpromotion.astype(str).str.lower()
        bad = ~v.isin(["true", "false", "nan", "none"])
        if bad.any():
            raise ValueError("Invalid onpromotion value")
        promo[row, col] = v.map({"true": 1.0, "false": 0.0}).to_numpy(np.float32)
    valid = np.array([np.asarray(dates >= first[k]) for k in keys])
    items = pd.read_csv(sources[1]).set_index("item_nbr", verify_integrity=True)
    stores = pd.read_csv(sources[2]).set_index("store_nbr", verify_integrity=True)
    oil = pd.read_csv(sources[3])
    oil["date"] = pd.to_datetime(oil.date)
    oil = (
        oil.set_index("date", verify_integrity=True)
        .dcoilwtico.reindex(dates)
        .ffill()
        .to_numpy(np.float32)
    )
    holidays = pd.read_csv(sources[4])
    holidays["date"] = pd.to_datetime(holidays.date)
    holidays = holidays[~holidays.transferred.astype(str).str.lower().eq("true")]
    holiday_by_store = {}
    static = []
    holiday = np.zeros((n, t), np.float32)
    for a, (s, i) in enumerate(keys):
        item = items.loc[i]
        store = stores.loc[s]
        static.append(
            {
                "store_nbr": str(s),
                "item_nbr": str(i),
                "family": str(item.family),
                "item_class": str(item["class"]),
                "perishable": str(item.perishable),
                "city": str(store.city),
                "state": str(store.state),
                "store_type": str(store.type),
                "cluster": str(store.cluster),
            }
        )
        if s not in holiday_by_store:
            h = holidays[
                (holidays.locale == "National")
                | (
                    (holidays.locale == "Regional")
                    & (holidays.locale_name == store.state)
                )
                | ((holidays.locale == "Local") & (holidays.locale_name == store.city))
            ]
            holiday_by_store[s] = dates.isin(h.date).astype(np.float32)
        holiday[a] = holiday_by_store[s]
    manifest = _save(
        out,
        ids,
        dates,
        y,
        valid,
        static,
        {"onpromotion": promo, "holiday": holiday},
        {"oil": oil},
        {
            "dataset": "favorita",
            "train_end": str(cut.date()),
            "seed": seed,
            "eligible_training_series": len(key_by_id),
            "source_hashes": {p.name: sha256(p) for p in sources},
            "negative_rows_in_date_range": neg,
            "scanned_rows": scanned,
            "returns_policy": returns,
            "zero_policy": "missing transaction after training-first-record -> zero; earlier dates masked; no inferred stockouts",
            "context_policy": "historical origin promotion only; oil forward-fill only; locale-matched non-transferred holidays; transactions excluded",
            "min_train_days": min_train_days,
        },
    )
    del y, promo
    (out / "_sales_work.npy").unlink()
    (out / "_promo_work.npy").unlink()
    write_json(
        out / "integrity.json",
        {
            p.name: sha256(p)
            for p in sorted(out.iterdir())
            if p.name != "integrity.json"
        },
    )
    return manifest


def load_retail_panel(cfg):
    path = Path(cfg["data"]["path"])
    meta = json.loads((path / "manifest.json").read_text())
    dates = pd.DatetimeIndex(meta["dates"])
    if cfg["data"].get("verify_hashes", True):
        for file, h in json.loads((path / "integrity.json").read_text()).items():
            if sha256(path / file) != h:
                raise ValueError(f"Prepared data integrity mismatch: {file}")
    stages, origins = make_stages(dates, cfg["split"], cfg["forecast"])
    if str(dates[stages["train"].end].date()) != meta["train_end"]:
        raise ValueError(
            "Training cutoff differs from cohort preparation; prepare again"
        )
    ids = meta["ids"]
    cap = cfg["data"].get("max_series")
    n = min(len(ids), cap) if cap else len(ids)
    ids = ids[:n]
    y = np.load(path / "y.npy", mmap_mode="r")[:n]
    valid = np.load(path / "valid.npy", mmap_mode="r")[:n]
    numeric = {
        k: np.load(path / f"context_{k}.npy", mmap_mode="r")[:n]
        for k in meta["numeric_columns"]
    }
    shared = {
        k: np.load(path / f"calendar_{k}.npy", mmap_mode="r")
        for k in meta["shared_columns"]
    }
    cfg["data"]["numeric_features"] = meta["numeric_columns"] + meta["shared_columns"]
    cfg["data"]["categorical_features"] = [
        k
        for k in meta["categorical_columns"]
        if k not in ["item_id", "item_nbr", "store_id", "store_nbr"]
    ]
    frames = [
        ContextFrame(
            meta["static"][i], {k: v[i] for k, v in numeric.items()}, shared, len(dates)
        )
        for i in range(n)
    ]
    audit = {k: v for k, v in meta.items() if k not in ["ids", "dates", "static"]}
    audit.update(
        {
            "sha256": content_hash(
                {"raw": meta["sha256"], "prepared": sha256(path / "manifest.json")}
            ),
            "retained_entities": n,
            "stage_dates": {
                s: [str(dates[v.start].date()), str(dates[v.end].date())]
                for s, v in stages.items()
            },
            "origins": {
                s: [str(dates[o].date()) for o in os] for s, os in origins.items()
            },
            "cohort_selection": "fixed training-only cohort; no evaluation coverage selection",
        }
    )
    return SalesPanel(ids, dates, y, valid, frames, stages, origins, audit)
