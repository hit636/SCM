"""Chronological fixed order-up-to sales-proxy replay; no oracle demand claim."""

from pathlib import Path
import numpy as np
import pandas as pd
from fame_scm.reporting import write_json


def bool_column(s, name):
    values = s.astype(str).str.lower()
    allowed = {
        "true": True,
        "false": False,
        "1": True,
        "0": False,
        "1.0": True,
        "0.0": False,
    }
    if not values.isin(allowed).all():
        raise ValueError(f"Invalid boolean {name}")
    return values.map(allowed).astype(bool)


def replay_inventory(predictions, daily, policy):
    """predictions: entity_id,origin,target_date,prediction.
    daily: entity_id,date,sales,available,uncensored,capacity,initial_stock.
    origin is end-of-day issue time. Orders placed after sales arrive morning
    date+lead_days. Target covers the next lead_days+review_days forecasts.
    Missing/insufficient issue-time forecasts fail; no future backfill.
    """
    p = predictions.copy()
    d = daily.copy()
    lead = int(policy["lead_days"])
    review = int(policy["review_days"])
    h = int(policy["forecast_horizon"])
    if lead < 1 or review < 1 or h < lead + review:
        raise ValueError("Invalid lead/review/horizon")
    safety = float(policy.get("safety_units", 0))
    under = float(policy.get("under_weight", 3))
    over = float(policy.get("over_weight", 1))
    if safety < 0 or under <= 0 or over <= 0:
        raise ValueError("Invalid policy costs/safety")
    for c in ["origin", "target_date"]:
        p[c] = pd.to_datetime(p[c], errors="raise").dt.normalize()
    d["date"] = pd.to_datetime(d["date"], errors="raise").dt.normalize()
    p["entity_id"] = p.entity_id.astype(str)
    d["entity_id"] = d.entity_id.astype(str)
    for c in ["available", "uncensored"]:
        d[c] = bool_column(d[c], c)
    for frame, cols in [
        (p, ["prediction"]),
        (d, ["sales", "capacity", "initial_stock"]),
    ]:
        for c in cols:
            frame[c] = pd.to_numeric(frame[c], errors="raise")
            if not np.isfinite(frame[c]).all() or (frame[c] < 0).any():
                raise ValueError(f"Invalid nonnegative {c}")
    if (
        p.duplicated(["entity_id", "origin", "target_date"]).any()
        or d.duplicated(["entity_id", "date"]).any()
    ):
        raise ValueError("Duplicate forecasts or daily rows")
    if (p.target_date <= p.origin).any():
        raise ValueError("Forecast target must be after issue origin")
    if set(p.entity_id) != set(d.entity_id):
        raise ValueError("Forecast and daily cohorts must match")
    records = []
    for entity, days in d.groupby("entity_id", sort=True):
        days = days.sort_values("date")
        dates = days.date.tolist()
        if (
            len(dates) > 1
            and not (days.date.diff().dropna() == pd.Timedelta(days=1)).all()
        ):
            raise ValueError(
                "Daily replay calendar must be contiguous, including unavailable days"
            )
        forecasts = p[p.entity_id == entity]
        issues = {
            o: g.set_index("target_date").prediction
            for o, g in forecasts.groupby("origin")
        }
        for o, f in issues.items():
            expected = pd.date_range(o + pd.Timedelta(days=1), periods=h)
            if not f.index.sort_values().equals(expected):
                raise ValueError(
                    "Every forecast issue must contain exactly the configured consecutive horizon"
                )
        stock = float(days.iloc[0].initial_stock)
        pipeline = []
        if stock > float(days.iloc[0].capacity):
            raise ValueError("Initial stock exceeds capacity")
        target = stock
        for k, row in enumerate(days.itertuples()):
            date = row.date
            before = stock
            arrivals = sum(q for due, q in pipeline if due <= date)
            pipeline = [(due, q) for due, q in pipeline if due > date]
            spilled = max(0.0, stock + arrivals - row.capacity)
            stock = min(row.capacity, stock + arrivals)
            sales_proxy = float(row.sales) if row.available else 0.0
            served = min(stock, sales_proxy)
            lost = sales_proxy - served
            stock -= served
            depletion = bool(
                row.available and row.uncensored and row.sales > 0 and stock <= 1e-10
            )
            # Score using a forecast issued strictly before this day's outcome.
            past = [o for o, f in issues.items() if o < date and date in f.index]
            if not past:
                raise ValueError(f"No causal forecast for {entity} on {date.date()}")
            yhat = float(issues[max(past)].loc[date])
            stock_loss = (
                (under * max(row.sales - yhat, 0) + over * max(yhat - row.sales, 0))
                if row.available
                else 0.0
            )
            order = 0.0
            issue_used = None
            if k % review == 0 and row.available:
                needed = pd.date_range(
                    date + pd.Timedelta(days=1), periods=lead + review
                )
                legal = [
                    o
                    for o, f in issues.items()
                    if o <= date and needed.isin(f.index).all()
                ]
                if not legal:
                    raise ValueError(
                        f"No issue-time forecast covering protection window for {entity} on {date.date()}"
                    )
                issue_used = max(legal)
                target = min(
                    row.capacity,
                    float(issues[issue_used].reindex(needed).sum()) + safety,
                )
                position = stock + sum(q for _, q in pipeline)
                order = max(0.0, target - position)
                if policy.get("integer_orders", False):
                    order = min(
                        np.ceil(order), max(0.0, np.floor(row.capacity - position))
                    )
                if order:
                    pipeline.append((date + pd.Timedelta(days=lead), order))
            records.append(
                {
                    "entity_id": entity,
                    "date": str(date.date()),
                    "opening_stock": before,
                    "arrivals": arrivals,
                    "capacity_discard": spilled,
                    "sales_proxy": sales_proxy,
                    "fulfilled_proxy": served,
                    "unfulfilled_proxy": lost,
                    "closing_stock": stock,
                    "order": order,
                    "in_transit": sum(q for _, q in pipeline),
                    "policy_target": target,
                    "excess_exposure": max(0.0, stock - target),
                    "depletion_event": depletion,
                    "eligible_depletion_day": bool(
                        row.available and row.uncensored and row.sales > 0
                    ),
                    "available": bool(row.available),
                    "causal_prediction": yhat,
                    "asymmetric_loss": stock_loss,
                    "order_forecast_origin": (
                        str(issue_used.date()) if issue_used is not None else None
                    ),
                }
            )
    r = pd.DataFrame(records)
    eligible = int(r.eligible_depletion_day.sum())
    valid = int(r.available.sum())
    result = {
        "policy": policy,
        "days": len(r),
        "available_days": valid,
        "eligible_depletion_days": eligible,
        "depletion_events": int(r.depletion_event.sum()),
        "depletion_rate": (
            float(r.depletion_event.sum() / eligible) if eligible else None
        ),
        "unfulfilled_sales_proxy": float(r.unfulfilled_proxy.sum()),
        "excess_inventory_exposure": float(r.excess_exposure.sum()),
        "asymmetric_loss_mean": (
            float(r.asymmetric_loss.sum() / valid) if valid else None
        ),
        "scope": "fixed-policy offline observed-sales proxy; not real stockout reduction or uncensored demand",
    }
    return r, result


def run_inventory(predictions_path, daily_path, policy, out):
    dest = Path(out)
    if dest.exists() and any(dest.iterdir()):
        raise ValueError("Replay output must be empty")
    rows, result = replay_inventory(
        pd.read_csv(predictions_path, dtype={"entity_id": str}),
        pd.read_csv(daily_path, dtype={"entity_id": str}),
        policy,
    )
    dest.mkdir(parents=True, exist_ok=True)
    rows.to_csv(dest / "inventory_daily.csv", index=False)
    write_json(dest / "inventory_summary.json", result)
    from fame_scm.reporting import content_hash
    import hashlib

    write_json(
        dest / "replay_manifest.json",
        {
            "predictions_sha256": hashlib.sha256(
                Path(predictions_path).read_bytes()
            ).hexdigest(),
            "daily_sha256": hashlib.sha256(Path(daily_path).read_bytes()).hexdigest(),
            "policy_hash": content_hash(policy),
        },
    )
    return result
