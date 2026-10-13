"""Estimate a frozen ranking noise margin from acquired rolling losses only."""

import numpy as np


def estimate_rank_margin(store, scaling, *, fallback=0.05, confidence=1.96):
    histories = {
        c.key: [store.cache[k].loss for k in c.requests] for c in store.observed_cells()
    }
    by_entity = {}
    for key in histories:
        by_entity.setdefault(key.entity_id, []).append(key)
    estimates = []
    for i, entity in enumerate(store.entities):
        columns = by_entity.get(entity, [])
        for a, ka in enumerate(columns):
            for kb in columns[a + 1 :]:
                difference = (
                    np.asarray(histories[ka]) - np.asarray(histories[kb])
                ) / scaling.scale[i]
                if len(difference) >= 2:
                    estimates.append(
                        float(np.std(difference, ddof=1) / np.sqrt(len(difference)))
                    )
    value = confidence * float(np.median(estimates)) if estimates else float(fallback)
    return value, {
        "value": value,
        "pair_histories": len(estimates),
        "definition": "1.96 times median standard error of paired normalized acquired origin losses",
        "fallback_used": not bool(estimates),
        "fallback": fallback,
        "scope": "acquired mining requests only; descriptive noise margin, not a pairwise significance test",
        "dependence": "overlapping origins can violate independence; configure disjoint horizons or use an author-supplied margin",
    }


def count_rank_pairs(labels, observed, margin, chunk_size=256):
    total = 0
    for start in range(0, len(labels), chunk_size):
        mask = observed[start : start + chunk_size]
        y = np.where(mask, labels[start : start + chunk_size], 0.0)
        total += int(
            (
                mask[:, :, None]
                & mask[:, None, :]
                & (y[:, :, None] + margin < y[:, None, :])
            ).sum()
        )
    return total
