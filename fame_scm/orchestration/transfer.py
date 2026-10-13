"""Identity holdout of competence/router training; backbone history remains legal."""

from copy import copy, deepcopy
from pathlib import Path
import json, pickle, time
import numpy as np
import pandas as pd
import torch
from fame_scm.datasets.snbc import SalesPanel
from fame_scm.fingerprints.features import Forecastability
from fame_scm.experts.forecast_experts import ExpertPool
from fame_scm.evaluation.costs import profile_costs
from fame_scm.datasets.loader import load_panel
from fame_scm.orchestration.experiment import run_method
from fame_scm.orchestration.execution import evaluate_stage, TimedProvider
from fame_scm.acquisition.evidence import MiningStore
from fame_scm.reporting import write_json, content_hash


def subset_panel(panel, indices):
    ix = np.asarray(indices, int)
    return SalesPanel(
        [panel.ids[i] for i in ix],
        panel.dates,
        panel.y[ix],
        panel.valid[ix],
        [panel.frames[i] for i in ix],
        panel.stages,
        panel.origins,
        {**panel.audit, "retained_entities": len(ix)},
    )


def identity_split(ids, component, fraction, seed):
    if not 0 < fraction < 1:
        raise ValueError("Holdout fraction must be in (0,1)")
    groups = [str(json.loads(x)[component]) for x in ids]
    unique = sorted(set(groups))
    if len(unique) < 2:
        raise ValueError("Need at least two distinct holdout identities")
    rng = np.random.default_rng(seed)
    held = set(
        rng.permutation(unique)[
            : max(1, min(len(unique) - 1, int(len(unique) * fraction)))
        ]
    )
    return np.array([i for i, g in enumerate(groups) if g not in held]), np.array(
        [i for i, g in enumerate(groups) if g in held]
    )


def run_holdout(cfg, component=1, fraction=0.2):
    out = Path(cfg["output"])
    if out.exists() and any(out.iterdir()):
        raise ValueError("Holdout output must be fresh")
    out.mkdir(parents=True)
    torch.set_num_threads(cfg["threads"])
    panel = load_panel(cfg)
    train_ix, test_ix = identity_split(panel.ids, component, fraction, cfg["seed"])
    train = subset_panel(panel, train_ix)
    test = subset_panel(panel, test_ix)
    feature = Forecastability(cfg).fit(train)
    start = time.perf_counter()
    pool = ExpertPool(cfg, panel).fit()
    preparation = time.perf_counter() - start
    pool.cost_profile = profile_costs(pool, panel, cfg)
    tp = copy(pool)
    tp.panel = train
    tp.fill = [pool.fill[i] for i in train_ix]
    hp = copy(pool)
    hp.panel = test
    hp.fill = [pool.fill[i] for i in test_ix]
    write_json(
        out / "holdout_protocol.json",
        {
            "training_entities": train.ids,
            "heldout_entities": test.ids,
            "component": component,
            "fraction": fraction,
            "seed": cfg["seed"],
            "shared_backbone_fit_seconds": preparation,
            "scope": "entire identities excluded from competence/router fitting and feature normalization; global backbones may use all legal training histories",
            "heldout_competence_labels_acquired": 0,
        },
    )
    results = []
    for mask in cfg["evaluation"]["mask_seeds"]:
        for seed in cfg["evaluation"]["neural_seeds"]:
            for method in cfg["baselines"]["methods"]:
                train_out = out / "training" / method / f"mask_{mask}" / f"seed_{seed}"
                summary = run_method(
                    cfg, train, tp, feature, method, mask, seed, train_out
                )
                router = None
                if method not in pool.names:
                    with (train_out / "deployment.pkl").open("rb") as f:
                        router = pickle.load(f)["router"].to(cfg["device"])
                store = MiningStore(
                    test.ids,
                    pool.names,
                    hp.availability(test.origins["mining"]),
                    0,
                    TimedProvider(test, hp),
                    {
                        k: content_hash({"name": k, "config": cfg["experts"]})
                        for k in pool.names
                    },
                    content_hash(test.ids),
                )
                metrics, records, routing = evaluate_stage(
                    "test",
                    test,
                    feature,
                    store,
                    router,
                    cfg,
                    summary["selected_threshold"],
                    fixed_expert=method if method in pool.names else None,
                )
                dest = out / "heldout" / method / f"mask_{mask}" / f"seed_{seed}"
                dest.mkdir(parents=True)
                pd.DataFrame(records).to_csv(dest / "predictions.csv", index=False)
                pd.DataFrame(store.ledger.events).to_csv(
                    dest / "execution_ledger.csv", index=False
                )
                write_json(dest / "routing.json", routing)
                write_json(dest / "metrics.json", metrics)
                results.append(
                    {"method": method, "mask": mask, "seed": seed, **metrics}
                )
    pd.DataFrame(results).to_csv(out / "heldout_results.csv", index=False)
    return results
