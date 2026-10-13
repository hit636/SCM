# Paper–code alignment

The supplied manuscript has 14 pages and the supplement six pages. Their exact
SHA-256 hashes are in [paper sources](../metadata/paper_sources.json). This file
records the implemented correspondence and the limits of that correspondence.

| Paper component | Code | Current implementation |
| --- | --- | --- |
| Mining evidence, Eq. (3) | `acquisition/evidence.py` | Charged seeds, unique cells, stage/origin/config-isolated cache, optional construction request cap |
| Fingerprints, Eq. (5) | `fingerprints/features.py` | Training-fitted statistics/vocabularies; winsorized slope, recency, recent mean change, spectral/lifecycle/context blocks |
| Competence, Eq. (6) | `competence/torch_models.py` | Two-layer shared encoder, row/expert biases, expert factors |
| Objectives, Eqs. (7)–(10) | `competence/torch_objectives.py`, `trainers.py` | Masked Huber, acquired-pair ranking, globally normalized minibatch terms, MSE ablation |
| Row scaling, Eq. (11) | `competence/scaling.py` | Frozen seed median/MAD; explicit training-only positive fallback; no-normalization control |
| Noise margin | `competence/calibration.py` | Acquired paired seed-origin SE heuristic; frozen once; fallback recorded |
| Acquisition, Eq. (12) | `acquisition/scores.py` | Restored-unit uncertainty/improvement/coverage/cost; four frozen batches |
| Teacher, Eqs. (13)–(15) | `routing/teacher.py` | Measured acquired entries; predicted missing entries plus uncertainty and cost |
| Routing, Eqs. (16)–(18) | `routing/gate.py` | Availability, Top-r, threshold pruning, nonempty fallback and fusion |
| Router objectives, Eqs. (19)–(24) | `routing/training.py` | Teacher KL + acquired cached mixture prediction + balance/cost |
| Stability, Eq. (29) | `evaluation/theory.py` | Post-selection component-loss bound terms, clipping tail, retained mass and nonvacuous fraction; fully executable rows only |
| Weighted work | `evaluation/audit.py`, `experiment.py` | Measured full mining reference after selection; acquisition uses frozen training profile |
| Five datasets | `datasets/`, `configs/experiments/` | Disjoint target stages, explicit origins, main horizons and 5×3 repetitions |
| Strong acquisition control | `baselines/recovery.py`, `experiment.py` | NN-EI/c: same shared neural Huber recovery, no ranking, expected improvement divided by frozen cost |
| Recovery controls | `baselines/neural.py`, `orchestration/suite.py` | Independent neural, shared Huber and shared Huber+rank; acquired-only labels |
| Consumer controls | `routing/consumers.py`, `orchestration/consumers.py` | Frozen common mask/teacher; direct teacher, KL-only, ST, Gumbel Top-r and Sparsemax |
| Teacher calibration | `orchestration/calibration.py` | Predeclared grid, calibration-only selection, all trials charged, selected test only |
| Scalability/pools | `orchestration/suite.py`, `experts/pools.py` | M=20 series-count sweep; nested 5/10/15/20/40 pools at supplement ratios |
| Row transfer | `orchestration/transfer.py` | Entire held-out identity groups excluded from recovery/router fitting |
| New column | `competence/insertion.py`, `orchestration/screening.py` | Appended-only factor/bias/head; exact old-logit/parameter checks; SCM/random/SoftImpute/full |
| Inference/statistics | `evaluation/statistics.py` | Paired records; 10K draws; positive D, centered two-sided p, conditional scope |
| Inventory | `evaluation/inventory.py` | Fixed-policy chronological replay and conservation tests |

## Explicit engineering choices

The ranking-noise heuristic, winsorized trend descriptor, training-scale fallback,
public cohort/cutoff schedules, expert hyperparameters, alternative router
operators, PMF-A adaptation, and pure-Torch selective SSM are recorded engineering
choices. The manuscript does not supply all original implementations or run
records; numerical equivalence to its code is not asserted.

The main teacher tau/eta/nu are fixed until explicit calibration search is run.
A search source checkpoint has already incurred its base fitting/calibration;
its account is reported separately from incremental search. Consumer controls
share a base recovery/teacher but have separately charged calibration. GPU work
cannot be inferred from wall time or CPU process hours.

Historical FAME D1–D3 selectors and every legacy C8 transfer comparator are not
implemented from unspecified rules. Real-data 100K performance, two-GPU resources,
short-lifecycle holdout results and operational replay require author protocols
and actual runs. Exact protocol matching rejects missing author records.

Paper target values remain isolated in `metadata/paper_targets.csv`; training and
measured export never ingest them. See [reproduction contract](reproduction.md),
[reviewer response](reviewer_response.md), and [validation evidence](validation.md).
