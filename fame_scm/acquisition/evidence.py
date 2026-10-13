from dataclasses import asdict
import numpy as np
from fame_scm.contracts import CellKey, CellEvidence, ForecastProvider, RequestKey


class ExecutionLedger:
    """Execution events and cache hits are recorded separately.
    This in-memory reference backend executes one request at a time.
    Cache hits are logged but do not create executed expert work.
    """

    def __init__(self, path=None):
        self.events = []
        self.path = str(path) if path else None
        self.context = {}

    def append(self, key, *, cache_hit, elapsed_seconds, success=True):
        self.events.append(
            {
                **asdict(key),
                "cache_hit": bool(cache_hit),
                "elapsed_seconds": float(elapsed_seconds),
                "success": bool(success),
                **self.context,
            }
        )
        if self.path:
            import json, time

            event = {**self.events[-1], "recorded_at": time.time()}
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(json.dumps(event) + "\n")

    def executed_seconds(self, stage):
        return sum(
            x["elapsed_seconds"]
            for x in self.events
            if x["stage"] == stage and not x["cache_hit"]
        )


class MiningStore:
    """Budget counts unique series/expert mining cells, including all seeds.
    The cache is separate from the learner-visible observations. A new trajectory
    must explicitly acquire a cached cell before its loss enters the evidence view.
    Unique cells are charged once; request caching is stage-specific.
    """

    def __init__(
        self,
        entities,
        experts,
        availability,
        budget,
        provider: ForecastProvider,
        config_hashes,
        protocol_hash,
        ledger=None,
        cache=None,
        construction_request_budget=None,
    ):
        self.entities, self.experts = tuple(entities), tuple(experts)
        if len(set(entities)) != len(entities) or len(set(experts)) != len(experts):
            raise ValueError("Duplicate entity/expert identifiers")
        self.availability = np.array(availability, bool, copy=True)
        if self.availability.shape != (len(entities), len(experts)):
            raise ValueError("Availability shape mismatch")
        self.budget = int(budget)
        if not 0 <= self.budget <= self.availability.sum():
            raise ValueError("Budget outside executable-cell count")
        self.provider = provider
        self.config_hashes = dict(config_hashes)
        if set(self.config_hashes) != set(experts):
            raise ValueError("Need a configuration hash for every expert")
        self.protocol_hash = protocol_hash
        self.ledger = ledger if ledger is not None else ExecutionLedger()
        self.cache = cache if cache is not None else {}
        self.construction_request_budget = construction_request_budget
        if construction_request_budget is not None and construction_request_budget < 0:
            raise ValueError("Construction request budget cannot be negative")
        self._construction_keys = set()
        self._cells = {}
        self._ei = {x: i for i, x in enumerate(entities)}
        self._mi = {x: m for m, x in enumerate(experts)}

    @property
    def remaining(self):
        return self.budget - len(self._cells)

    def acquire(self, entity_id, expert_id, origins):
        cell = CellKey(entity_id, expert_id)
        i, m = self._ei[entity_id], self._mi[expert_id]
        if not self.availability[i, m]:
            raise ValueError("Unavailable cell cannot be acquired")
        if len(origins) == 0 or len(set(origins)) != len(origins):
            raise ValueError("Need a nonempty unique origin list")
        keys = tuple(
            RequestKey(
                "mining",
                entity_id,
                expert_id,
                str(o),
                self.config_hashes[expert_id],
                self.protocol_hash,
            )
            for o in origins
        )
        if cell in self._cells:
            if self._cells[cell].requests != keys:
                raise ValueError("Cell origins changed within a frozen protocol")
            return self._cells[cell]
        if self.remaining == 0:
            raise ValueError("Mining cell budget exhausted")
        responses = [self.fetch(k) for k in keys]
        # Paper: average stage-origin expert losses. Target-level pooled forecast
        # metrics are computed separately; do not interchange the two aggregations.
        evidence = CellEvidence(cell, keys, float(np.mean([x.loss for x in responses])))
        self._cells[cell] = evidence
        return evidence

    def fetch(self, key):
        """Infrastructure request path, not an API to hand to a learner."""
        if (
            key.stage in ["mining", "calibration"]
            and key not in self._construction_keys
        ):
            if (
                self.construction_request_budget is not None
                and len(self._construction_keys) >= self.construction_request_budget
            ):
                raise ValueError(
                    "Total construction request budget exhausted (mining + calibration)"
                )
            self._construction_keys.add(key)
        if key in self.cache:
            self.ledger.append(key, cache_hit=True, elapsed_seconds=0)
            return self.cache[key]
        import time

        started = time.perf_counter()
        try:
            result = self.provider(key)
            if result.key != key or not np.isfinite(result.loss):
                raise ValueError("Invalid provider evidence")
            if not np.isfinite(result.prediction).all() or result.elapsed_seconds < 0:
                raise ValueError("Invalid provider prediction/time")
        except Exception:
            self.ledger.append(
                key,
                cache_hit=False,
                elapsed_seconds=time.perf_counter() - started,
                success=False,
            )
            raise
        self.cache[key] = result
        self.ledger.append(key, cache_hit=False, elapsed_seconds=result.elapsed_seconds)
        return result

    def observed_cells(self):
        return tuple(self._cells.values())

    def snapshot(self):
        losses = np.full(self.availability.shape, np.nan)
        mask = np.zeros(self.availability.shape, bool)
        for cell, evidence in self._cells.items():
            i, m = self._ei[cell.entity_id], self._mi[cell.expert_id]
            losses[i, m], mask[i, m] = evidence.loss, True
        return losses, mask

    def cached_observed_forecasts(self):
        return {
            cell.key: tuple(self.cache[k] for k in cell.requests)
            for cell in self.observed_cells()
        }

    def evidence_account(self):
        stages = {}
        for stage in ["mining", "calibration", "test", "research_audit"]:
            events = [e for e in self.ledger.events if e["stage"] == stage]
            requests = {
                (
                    e["entity_id"],
                    e["expert_id"],
                    e["origin"],
                    e["config_hash"],
                    e["protocol_hash"],
                )
                for e in events
                if e["success"]
            }
            stages[stage] = {
                "unique_origin_requests": len(requests),
                "unique_cells": len({(k[0], k[1]) for k in requests}),
                "executed_events": sum(not e["cache_hit"] for e in events),
                "cache_hits": sum(e["cache_hit"] for e in events),
                "failed_events": sum(not e["success"] for e in events),
                "executed_seconds": self.ledger.executed_seconds(stage),
            }
        return {
            "stages": stages,
            "construction_unique_origin_requests": len(self._construction_keys),
            "construction_request_budget": self.construction_request_budget,
            "construction_stage_cells_sum": stages["mining"]["unique_cells"]
            + stages["calibration"]["unique_cells"],
            "scope": "Stage-origin requests are distinct; calibration is never merged into mining rho; cache reuse does not bypass the evidence cap",
        }


class AcquiredEvidenceView:
    """Expose only this view to recovery/trainers; no fetch or shared cache API."""

    def __init__(self, store):
        self.__store = store

    def observed_cells(self):
        return self.__store.observed_cells()

    def snapshot(self):
        return self.__store.snapshot()


def weighted_work_ratio(observed, availability, offline_costs):
    observed, availability = np.asarray(observed, bool), np.asarray(availability, bool)
    costs = np.broadcast_to(np.asarray(offline_costs, float), availability.shape)
    if observed.shape != availability.shape or np.any(observed & ~availability):
        raise ValueError("Observed cells must be executable")
    if not np.isfinite(costs[availability]).all() or np.any(costs[availability] <= 0):
        raise ValueError("Executable costs must be positive finite")
    denominator = costs[availability].sum()
    if denominator <= 0:
        raise ValueError("Empty full-work reference")
    return float(costs[observed].sum() / denominator)
