# Engineering response to reproducibility concerns

This is a code/evidence response for preparing a reviewable submission. It does
not state that a journal reviewer has accepted the paper or that synthetic tests
establish its empirical conclusions.

| Concern | Revision | Reviewable evidence |
| --- | --- | --- |
| A 40% cell budget hides calibration or repeated origins | Distinct mining/calibration/test/audit request accounting; charged seeds; optional total construction cap | `evidence_account`, execution ledger, budget contract tests, `total_budget` suite |
| Acquisition lacks a strong neural cost-aware comparator | Executable NN-EI/c with the shared neural Huber ensemble and ranking disabled | Same pool/cohort/protocol hashes; common prediction audit |
| Recovery sharing is confounded with architecture | Independent 2×128 neural recovery, shared Huber and shared Huber+rank jobs | `recovery` profile; masks/loss labels explicitly acquired only |
| Objective ablations are incomplete | No-Huber/MSE and no-row-normalization added alongside ranking and acquisition controls | `ablations` plan; objective/scaling tests |
| Teacher/consumer comparison changes evidence | Six consumers use one frozen Omega and teacher | Consumer mask and teacher hashes asserted identical |
| Teacher hyperparameters are selected using test results | Explicit calibration-only search; only selected trial touches test | Per-trial events, search log, shared-source account, selected test ledger |
| Ranking margin is unexplained | Acquired seed-origin paired SE estimate, frozen with recorded fallback | `rank_margin.json`; noise dependence caveat and acquired-cache tests |
| Training minibatches reweight rows/pairs accidentally | Reconstruction/ranking use global observed-count normalizers; pair counting chunked | Objective tests; bounded-memory pair-count implementation |
| Cached or incomplete work is called a successful experiment | Stage/config cache keys; failed-run artifact; completed-job hash; hard-deadline complete-checkpoint selection | Cache isolation, atomic checkpoint, deadline validator |
| Weighted work ignores row heterogeneity | Dense measured full mining reference after choices freeze; insertion full-column reference | Audit cost matrix and weighted-ratio recomputation |
| Hidden labels leak into fitting | Dense losses queried only after acquisition, calibration and predictions freeze | Acquired-only objective/ranking tests; evaluator-only modules |
| Claimed statistical significance is unsupported | 10K paired entity/time draws, centered two-sided p and signed D | Paired prediction hashes and comparison JSON; no fabricated p-values |
| Theory does not imply fused forecast performance | Report component-loss bound terms, discarded mass, clipping tail and nonvacuous fraction | `theory_rows.csv`, `theory_summary.json`; excludes unavailable columns and states scope |
| New expert update retrains old experts | Only appended factor/bias/head optimized; old modules/logits checked exactly | Five real inserted backends × ten configurations passed |
| Optional experts are silent placeholders | Real backends, pinned model revision, failures explicit; compatibility pins for PyArrow/Transformers | All 40 configured candidates checked for finite shape and save/reload equality |
| Resource claims cannot be recreated | Measured process CPU, sampled RSS with explicit unavailable values, separate stages/preparation; GPU hours null | Resource scope and sampling error fields; no invented two-GPU result |
| Artifact is not a maintainable repository | Installable package, optional extras, complete environment record, CI, unit/integration validators, provenance and docs | Build/install/CLI and repository checks |

## What still requires author experiments

A fully reviewable empirical revision must attach the actual main-run predictions,
5×3 seed records, calibration trials, total-budget comparisons, complete data
protocol and real resource telemetry. It must run the 100K jobs on the intended
hardware and evaluate the declared inventory inputs/policy. Original proprietary
SNBC, exact public cohorts/origins, legacy selector definitions, and original
operational records were not supplied. The repository exposes this boundary and
rejects exact-paper labels without an external matching protocol; it does not
fill gaps with the published scores.

The public split defaults and several controls are engineering specifications.
Report them as such unless matched to author records. A fixed 0.05 fallback margin
and the optional 243-fit calibration grid are not asserted to be the manuscript's
unpublished selection procedure. Conditional bootstrap intervals are not evidence
of adaptive-neural sample complexity. See [reproduction](reproduction.md).
