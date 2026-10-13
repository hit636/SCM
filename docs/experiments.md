# Experiment protocols

## Chronological access

Training fits forecasting backbones and feature statistics. Mining acquires
competence evidence and cached forecasts. Calibration evaluates threshold
choices at separate origins. Test forecasts are frozen before evaluator-only
hidden-cell diagnostics.

A stage origin is admissible only when its complete target horizon lies within
that stage. Lookbacks may use all earlier legal observations. Global calendar
cutoffs apply to every entity; per-series target masks remain explicit.

Main configurations use five evidence seeds and three neural seeds. Forecast
models use a shared training seed within an experiment. Each selector receives
its own acquired evidence and writes its own execution ledger.

## Selection methods

| Identifier | Recovery and acquisition |
| --- | --- |
| `lightgbm` | Single fitted anchor expert |
| `full_observation` | All executable mining cells and the same router objective |
| `random_partial` | Nearest-neighbor imputation from random acquired evidence |
| `softimpute` | Singular-value shrinkage on acquired normalized losses |
| `imc` | Fixed-fingerprint bilinear factors fitted on acquired entries |
| `independent_lgbm` | One loss regressor per expert |
| `fformpp_b` | Complete-row acquisition and pinned movingknots error prediction |
| `independent_neural` | Independent nonlinear acquired-loss regressors |
| `nn_ei_c` | Shared neural Huber ensemble, no ranking, expected improvement per offline cost |
| `pmf_a` | GP latent expert embeddings, posterior conditioning and expected improvement |
| `scm_passive` | Shared Huber/ranking competence ensemble with random remainder acquisition |
| `fame_scm_active` | Shared Huber/ranking ensemble with Eq. (12) acquisition |

All methods within an experiment use the same forecast backbones, executable
pool, origin sets, masks, and pooled metric definitions. Cell budgets include
initialization. FFORMPP-B spends residual budget on a partial row but uses
complete rows for joint fitting.

## Suite profiles

| Profile | Jobs |
| --- | --- |
| `budgets` | rho 0.2, 0.4, 0.6, 0.8; each includes its configured reference methods |
| `mechanism` | Ranking weight 0 or 0.2 crossed with passive or active acquisition |
| `ablations` | Full method, no Huber, no normalization, no ranking, no fingerprints, no uncertainty penalty, no balance/cost, KL-only, no teacher KL, no coverage, no acquisition cost, uncertainty-only, improvement-only, uncertainty plus improvement |
| `sensitivity` | Alpha 0.5/1/2; beta 0.1/0.5/1; Top-r 1/2/3; teacher tau and eta |

Job settings are fixed before execution and written as YAML. Completed jobs have
a configuration-hash completion record. Cost profiles are shared across jobs
when the data/pool signature is unchanged; scale/pool jobs get separate profiles. Acquisition changes generate a separate
trajectory and incur their own execution and recovery costs.

## Transfer and insertion

Identity components are indices of the JSON entity-ID tuple. For retail panels,
component 0 is the store and component 1 the item. Holdout excludes complete
groups from competence/router fitting and feature-statistic estimation.

Insertion starts from a calibrated SCM research checkpoint. The shared old
expert cache and parameters are common preparation. The new expert is fitted
independently, and acquired new-column labels update only its embedding/bias
and appended router head. Per-column screening ratios are separate from the
old-pool mining ratio. Update time, backbone fitting, stage execution, use rate,
and error relative to old/full-inserted references are recorded.

## Equal construction time

Hard-deadline workers attempt increasing mining ratios in the declared order.
Every attempt restarts its mining and training trajectory. The latest completed
calibrated checkpoint at the deadline is selected without reading test scores.
The scoring phase runs after construction and is excluded from its deadline.

Preparation scope is explicitly selected. Shared preparation includes data,
fingerprints, backbones, and cost profiling. With `shared_excluded`, preparation
is saved once and charged separately; checkpoint loading and repeated builds
remain inside each construction deadline. CPU RSS is sampled across worker
processes. The backend records wall time and process memory, rather than GPU
resource-hours.

## Statistical comparisons

`compare` verifies paired prediction records before computing a two-way
entity-group by target-date-block bootstrap. Each sample recomputes the chosen
pooled metric, averages neural-seed metrics within masks, and averages mask
results. Positive `reference_minus_method` favors the selected method.

At least two entity groups and two time blocks are required. The interval is
conditional on fitted models and recorded evidence trajectories. Across-mask
standard deviations in aggregate tables summarize a different source of
variation. Rolling origins and overlapping target records are not independent
experimental repetitions.

## Inventory replay

The replay uses the same fixed policy for every forecast substitution. A
one-day lead time, explicit capacity, initial stock, uncensored sales proxy,
and availability masks define the daily state transitions. Conservation checks
cover opening stock, arrivals, discarded capacity excess, fulfilled sales, and
closing stock. Reported depletion events use positive uncensored observed
sales and zero closing stock. Replay effects apply to the declared proxy and
policy, rather than an intervention on latent uncensored demand.

## Additional plans and controls

`recovery` supplies independent neural / shared Huber / shared Huber+rank jobs.
`pools` uses Supplement C7 ratios for 5/10/15/20/40 candidates. `scale` uses
5K/10K/25K/50K/100K requested rows and M=20/rho=0.305. Input cohorts must actually
support those counts. `total_budget` requires explicitly supplied
`budgets.comparison_requests`; no absolute budget is inferred from rho.

`consumers` starts from the same completed SCM build checkpoint for all six
operators, asserts identical observed-mask/teacher hashes, and charges separate
calibration. `calibrate` runs fixed or explicit grid settings only on calibration
and tests its selected trial. Both report shared-source evidence separately.
Repeating either experiment requires the matching checkpoint per mask/seed.
Insertion repetition also requires a `{mask}`/`{seed}` checkpoint path template.

Measured exports read successful `summary.json` files only; paper target metadata
is never consulted. Use separate output roots for distinct experiments. Repeated
controls must have matching protocol metadata before numerical comparisons.
