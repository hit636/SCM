# Output schema

## Experiment root

| File | Content |
| --- | --- |
| `resolved_config.yaml` | Fully merged input and experiment configuration |
| `data_audit.json` | Input hashes, data policies, cohort and chronological boundaries |
| `cohort.json` | Ordered entity IDs and hash |
| `features.json` | Training vocabulary, feature means/scales and frozen fill values |
| `cost_profile.json` | Expert weights, profiling requests, elapsed profiling time and signature |
| `reproduction_protocol.json` | Input/cohort/origin/selector/evaluation binding and exact-reproduction status |
| `run_manifest.json` | Software/hardware information, code/config hashes and common preparation times |
| `results.csv` | Per-method, mask and neural-seed metrics |
| `aggregate.csv` | Means and standard deviations of within-mask seed averages |
| `expert_checkpoints/` | Fitted candidate model states and training records |
| `consistency_audit.json` | Recomputed metrics and cross-method invariant checks |

## Method / mask / seed directory

Directories use `<method>/mask_<id>/seed_<id>/`.

| File | Content |
| --- | --- |
| `summary.json` | Metrics, diagnostics, mining budget, evidence counts and elapsed stages |
| `predictions.csv` | Entity, origin, horizon step, target date, target, prediction, validity and weight |
| `execution_ledger.csv` | Request keys, cache-hit status, success and actual execution seconds |
| `execution_events.jsonl` | Append-only execution events with wall timestamps |
| `evidence_cells.csv` | Acquired entity/expert cells, average losses and origin lists |
| `initialization.json` | Charged seed cells and fixed seed policy |
| `rank_margin.json` | Frozen acquired-noise margin estimate and fallback |
| `scaling.json` | Frozen seed-derived row centers/scales and training fallback specification |
| `competence.npz` | Availability, acquired mask/losses, recovered means, uncertainty and teacher |
| `acquisition_trace.json` | Frozen-batch acquisitions and cumulative cell counts |
| `routing.json` | Dense probabilities, pre-pruning Top-r, selected experts and fusion weights |
| `calibration_trials.json` | Threshold trials and primary selection metric |
| `competence_training.csv` | Final ensemble acquired-objective training history |
| `router_training.csv` | Router objective history |
| `router.pt`, `competence_<member>.pt` | Neural parameter states |
| `deployment.pkl` | Compact fitted forecasting bundle |
| `build_state.pkl` | Optional research state for insertion or construction-time evaluation |
| `evaluator_validation_losses.npy` | Post-selection evaluator-only dense mining losses |
| `evaluator_cell_work_seconds.npy` | Selected/hidden-audit per-cell measured work reference |
| `theory_rows.csv`, `theory_summary.json` | Component-loss stability terms, tail and nonvacuous fraction |
| `row_evidence_diagnostics.csv` | Per-row acquired count, singleton status and retained mass |
| `failed_run.json` | Explicit failure, never a completed benchmark |

Single-expert references omit competence/router artifacts.

## Summary fields and cost boundaries

`actual_rho` equals unique acquired mining cells divided by executable mining
cells. `weighted_cost_ratio` identifies its reference in `weighted_cost_reference`: main configs use measured selected/hidden-audit per-cell work, while smoke configs may use frozen training-profile weights.
`evidence_cells_by_stage` reports unique successful executed entity/expert pairs
for mining, calibration, test, and research audit separately.
`construction_distinct_series_expert_pairs` counts the union of mining and
calibration pairs. These counts are distinct from per-origin execution requests.

`seconds.build` sums mining, router and calibration elapsed time. Shared expert
fitting, feature/data preparation and cost profiling remain in the root manifest.
`seconds.test` and `seconds.research_audit` describe evaluation work.

`hit_at_2_validation` evaluates the best validation expert's presence before
threshold pruning. `execution_count_mean` counts the retained set after pruning.
`retained_probability_mass_mean` records retained available probability mass.
`recovery_mae_hidden` uses each policy's unobserved complement; acquisition
policies can change the difficulty of that complement.

Raw retail/daily metrics and train-zscore hourly metrics are identified by
`metric_scale`. WAPE and sMAPE are ratios. Undefined WAPE for a zero target
absolute-sum denominator is represented by JSON null.

## Specialized experiments

| Experiment | Main outputs |
| --- | --- |
| Suite | Job YAMLs, immutable plan, per-job completion hashes and search-cost summary |
| Holdout | Identity protocol, held-out predictions/ledgers and `heldout_results.csv` |
| Insertion | Per-strategy column traces, predictions, frozen-module checks, use rates and `insertion_results.json` |
| Equal time | Worker logs, completed checkpoints, test predictions and `equal_time_results.json` |
| Paired comparison | `comparison_<method>_vs_<reference>_<metric>.json` |
| Inventory | Chronological replay records and fixed-policy summary |

JSON and research checkpoints use atomic replacement. Generated artifacts live
under the chosen output directory and are excluded from Git.

## New account and telemetry fields

`evidence_account.stages` contains unique origin requests, unique pairs,
`executed_events`, cache hits, failures and executed seconds for each stage.
`construction_unique_origin_requests` counts unique mining+calibration keys
across source and current caches; `construction_request_budget` is its optional
cap. This is different from physical execution events and mining rho.
Consumer/search summaries also contain `shared_source_evidence_account`; source
preparation is not duplicated as newly executed work in their physical ledgers.

`resources` records wall seconds, process CPU seconds/hours, sampled process-tree
RSS, sampling interval, scope and a memory sampling error. RSS is null if no
valid sample was available. GPU resource hours are null without external telemetry.

`calibration_search.json` records every teacher/router/threshold trial.
`consumer_results.json` includes shared mask/teacher hashes.
`full_column_reference_work.json` records post-selection new-column denominator
weights. `tables` writes measured run/aggregate/evidence CSVs and an export manifest;
aggregate means and sample SD use neural-seed means within evidence masks.
