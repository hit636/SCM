"""Future forecasting from a saved local deployment bundle."""

from pathlib import Path
import pickle
import numpy as np
import pandas as pd
import torch
from fame_scm.datasets.snbc import read_sales, SalesPanel, CATEGORICAL, NUMERIC
from fame_scm.routing.gate import tkde_gate, fuse
from fame_scm.datasets.temporal import time_step, timestamp_text
from fame_scm.routing.training import router_probabilities


def predict_file(checkpoint, data_path, output_path, *, device=None):
    with open(checkpoint, "rb") as file:
        bundle = pickle.load(file)
    cfg = bundle["config"]
    from fame_scm.devices import prepare_device

    selected_device = prepare_device(device if device is not None else cfg["device"])
    cfg["device"] = str(selected_device)
    cfg["data"]["path"] = data_path
    torch.set_num_threads(cfg["threads"])
    if cfg["data"].get("format") == "wide_csv":
        from fame_scm.datasets.wide import load_wide_panel

        panel = load_wide_panel(cfg, inference=True, scaling=bundle.get("data_scalers"))
        ids = panel.ids
        dates = panel.dates
    elif cfg["data"].get("format") == "retail_panel":
        from fame_scm.datasets.retail import load_retail_panel

        panel = load_retail_panel(cfg)
        ids = panel.ids
        dates = panel.dates
    else:
        frame, audit = read_sales(cfg)
        dates = pd.date_range(frame["_date"].min(), frame["_date"].max())
        ids = []
        ys = []
        valid = []
        frames = []
        for entity, g in frame.groupby("_id", sort=True):
            g = g.set_index("_date").reindex(dates)
            seen = g["_y"].notna().to_numpy()
            mask = seen.copy()
            if cfg["data"]["missing_dates"] == "zero":
                mask[np.flatnonzero(seen)[0] :] = True
            col = cfg["data"].get("availability_column")
            if col:
                if col not in g:
                    raise ValueError("Missing configured availability column")
                mask &= (
                    g[col]
                    .fillna("0")
                    .astype(str)
                    .isin(["1", "true", "True"])
                    .to_numpy()
                )
            for c in CATEGORICAL + NUMERIC:
                if c not in g:
                    g[c] = np.nan
                g[c] = g[c].ffill()
            if mask.sum() < cfg["forecast"]["lookback"]:
                continue
            ids.append(entity)
            ys.append(g["_y"].fillna(0).to_numpy(float))
            valid.append(mask)
            frames.append(g)
        if not ids:
            raise ValueError("Insufficient history for future prediction")
        panel = SalesPanel(
            ids, dates, np.array(ys), np.array(valid), frames, {}, {}, audit
        )
    pool = bundle["expert_pool"]
    old_fill = dict(
        zip(pool.training_ids if pool.panel is None else pool.panel.ids, pool.fill)
    )
    global_fill = float(np.median(pool.fill))
    pool.panel = panel
    pool.fill = [old_fill.get(i, global_fill) for i in ids]
    pool.to(selected_device)
    router = bundle["router"].to(selected_device).eval()
    from fame_scm.routing.consumers import DirectTeacher

    if isinstance(router, DirectTeacher) and router.entities != list(ids):
        raise ValueError(
            "Direct-teacher control requires the original frozen row identities and ordering; use the distilled router for new entities"
        )
    origin = len(dates) - 1
    z = bundle["features"].transform(panel, origin)
    p = router_probabilities(router, z)
    gate = tkde_gate(
        p,
        pool.availability([origin]),
        r=cfg["router"]["r"],
        threshold=bundle["threshold"],
    )
    cache = np.full((len(ids), len(pool.names), pool.h), np.nan)
    for i, j in zip(*np.where(gate.retained)):
        cache[i, j] = pool.predict(pool.names[j], i, origin)
    prediction = fuse(gate.weights, cache)
    rows = []
    if cfg["data"].get("format") == "wide_csv":
        for i, entity in enumerate(ids):
            transform = panel.audit["scalers"][entity]
            prediction[i] = prediction[i] * transform["scale"] + transform["mean"]
    for i, entity in enumerate(ids):
        for h in range(pool.h):
            rows.append(
                {
                    "entity_id": entity,
                    "origin": timestamp_text(dates[-1], cfg),
                    "date": timestamp_text(dates[-1] + (h + 1) * time_step(cfg), cfg),
                    "forecast_quantity": float(prediction[i, h]),
                    "experts": ",".join(
                        pool.names[j] for j in np.flatnonzero(gate.retained[i])
                    ),
                }
            )
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8-sig")
    return str(path)
