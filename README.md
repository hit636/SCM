<!-- section: overview -->

# FAME-SCM

Official implementation of **Sparse Competence Mining for Forecastability-Aware Expert Routing in Heterogeneous Time Series**.

FAME-SCM learns forecastability-aware expert routing from sparsely acquired series–expert competence evidence. It combines training-time fingerprints, shared competence recovery, cost-aware active acquisition, teacher-guided sparse routing, and incremental expert insertion. The repository provides configurations and experiment entry points for the paper and supplementary analyses.

<!-- section: method -->

## Method and code structure

| Paper component | Implementation |
| --- | --- |
| Forecastability fingerprints, Eq. (5) | `fame_scm/fingerprints/` |
| Shared competence model and objectives, Eqs. (6)–(11) | `fame_scm/competence/` |
| Sparse evidence and active acquisition, Eqs. (3), (12) | `fame_scm/acquisition/` |
| Competence teacher and sparse routing, Eqs. (13)–(24) | `fame_scm/routing/` |
| Forecasting experts and selection baselines | `fame_scm/experts/`, `fame_scm/baselines/` |
| Metrics, work accounting, statistical comparisons and inventory replay | `fame_scm/evaluation/` |
| Training, calibration and experiment suites | `fame_scm/orchestration/` |

The pipeline uses four chronological stages: **training → mining → calibration → testing**. Training fits expert backbones and fingerprint statistics; mining acquires competence evidence; calibration selects routing settings; testing exports forecasts and evaluation records.

<!-- section: installation -->

## Environment setup

Use Python 3.10–3.12 and run the commands from the repository root. Windows users can use Git Bash. The project runs directly from the local `fame_scm/` package. Install third-party libraries through the single `requirements.txt`.

Create and activate an environment:

```bash
python -m venv .venv
# Linux/macOS:
source .venv/bin/activate
# Windows Git Bash: use source .venv/Scripts/activate instead.
```

<!-- section: devices -->

### GPU and CPU

For NVIDIA GPU experiments, install the CUDA build of Torch and check the selected device. The command below uses the CUDA 12.4 wheel from the [official PyTorch installation instructions](https://pytorch.org/get-started/previous-versions/).

```bash
python -m pip install --upgrade pip
python -m pip install torch==2.6.0+cu124 --index-url https://download.pytorch.org/whl/cu124
python -m pip install -r requirements.txt
python scripts/check_device.py --device cuda:0
```

<!-- section: cpu_setup -->

For a CPU environment, use this alternative installation block:

```bash
python -m pip install torch==2.6.0+cpu --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
python scripts/check_device.py --device cpu
```

<!-- section: device_policy -->

Main experiment configurations select `cuda:0`. Use `--device cuda:N` to select a visible GPU, or `--device cpu` for CPU execution. Each job uses one selected device; independent jobs can run on separate GPUs with separate output directories. A GPU run requires a compatible NVIDIA driver and a visible CUDA device.

SCM networks, neural routers, neural experts, and the Torch fitting components of IMC/GP recovery use the selected device. Statistical experts, tree models, feature processing and metrics use CPU. FFORMPP-B uses the supplied JAX CPU dependencies. Chronos loads the checkpoint revision configured in the expert pool.

<!-- section: quickstart -->

## Quick start

Run the encapsulated API example on generated synthetic daily series. The Bash entry point prepares data, trains two methods, evaluates forecasts and audits the outputs. Its output directory contains `synthetic_daily.csv`, `experiment/` and `api_validation.json`.

```bash
bash scripts/run_api.sh --out runs/api_example --device cuda:0
python run.py --help
```

<!-- section: data -->

## Datasets and preparation

| Dataset | Input location | Main configuration | Forecast horizon | Primary metric |
| --- | --- | --- | --- | --- |
| SNBC | `data/snbc/daily_sales_data.csv` | `configs/experiments/snbc_main.yaml` | 14 days | RMSE |
| M5 | `data/m5_prepared/` | `configs/experiments/m5_main.yaml` | 28 days | WAPE |
| Favorita | `data/favorita_prepared/` | `configs/experiments/favorita_main.yaml` | 28 days | WAPE |
| Electricity | `data/electricity/electricity.csv` | `configs/experiments/electricity_main.yaml` | 168 hours | MSE |
| Traffic | `data/traffic/traffic.csv` | `configs/experiments/traffic_main.yaml` | 168 hours | MSE |

Place the datasets at the configured paths, or provide their locations through the dataset runner. SNBC uses a daily-sales CSV with `date`, `vem_id`, `merc_id` and `daily_quantity`. Electricity and Traffic use timestamped wide CSV files. Hourly standardization uses training-fitted per-series statistics; exported forecasts use the original units.

Prepare M5 and Favorita from their raw dataset folders. Keep the preparation cutoff aligned with the first split cutoff in the dataset configuration. See [data formats](docs/data_formats.md) for field definitions and data policies.

```bash
python run.py prepare-retail --dataset m5 --raw data/m5_raw --out data/m5_prepared --train-end 2014-10-17
python run.py prepare-retail --dataset favorita --raw data/favorita_raw --out data/favorita_prepared --train-end 2016-03-26
```

<!-- section: main_experiments -->

## Main experiments

Main configurations use a **40% mining-cell observation budget** and **five evidence-mask seeds × three neural seeds**. Methods share the dataset cohort, chronological splits, candidate pool and fitted forecasting experts. Each selection method acquires its own evidence and writes its execution ledger.

Run a single dataset:

```bash
python run.py doctor --config configs/experiments/snbc_main.yaml --device cuda:0 --check-data
python run.py run --config configs/experiments/snbc_main.yaml --device cuda:0 --out runs/snbc_main
python run.py audit --out runs/snbc_main
```

<!-- section: dataset_runner -->

Run all five datasets with the main profile. Add `--dry-run` to inspect the generated configurations. Dataset paths can be overridden with `--snbc-data`, `--m5-data`, `--favorita-data`, `--electricity-data` and `--traffic-data`.

```bash
python scripts/run_datasets.py --profile main --device cuda:0 --out runs/five_datasets
```

<!-- section: baselines -->

### Selection methods

| Method | Configuration identifier |
| --- | --- |
| LightGBM anchor | `lightgbm` |
| Full observation | `full_observation` |
| Random Partial | `random_partial` |
| SoftImpute | `softimpute` |
| Fixed-feature IMC | `imc` |
| Independent regression | `independent_lgbm` |
| FFORMPP-B | `fformpp_b` |
| PMF-A | `pmf_a` |
| NN-EI/c | `nn_ei_c` |
| Passive SCM | `scm_passive` |
| Active SCM | `fame_scm_active` |

Use `--methods` on `run.py run` to select a subset. Independent neural recovery is available through the recovery suite.

<!-- section: experiment_map -->

## Paper experiment map

| Paper reference | Experiment | Entry point |
| --- | --- | --- |
| Table V | Five-dataset forecasting accuracy and evaluation work | Main configurations / `run_datasets.py` |
| Table VI | Paired comparisons and confidence intervals | `compare` |
| Table VII | Equal construction-time comparison | `equal-time` |
| Table VIII; Tables C5–C6 | Component ablations and teacher-to-router controls | `suite --profile ablations`, `consumers` |
| Fig. 3; Table C1 | Mining-budget analysis | `suite --profile budgets` |
| Fig. 4; Tables C2–C4 | Recovery/acquisition mechanisms and recovery architecture | `suite --profile mechanism`, `suite --profile recovery` |
| Tables IX–X; Fig. 5 | Scale and construction-resource analysis | `suite --profile scale` |
| Tables C5–C6 | Parameter sensitivity and routing operators | `suite --profile sensitivity`, `consumers` |
| Table C7 | Candidate-pool analysis | `suite --profile pools` |
| Table XI(a) | Unseen-identity transfer | `holdout` |
| Table XI(b); Table C9; Fig. 6 | New-expert insertion and screening | `insertion`, `extended_suite.py insertion` |
| Table D4 | Fixed-policy inventory replay | `inventory` |

Experiment configurations, output schemas and detailed protocols are documented in [experiments](docs/experiments.md) and [output schema](docs/output_schema.md).

<!-- section: ablation -->

## Ablations, budgets and sensitivity

The suites cover observation budgets, ranking/acquisition mechanisms, Huber loss, row normalization, fingerprints, uncertainty, coverage, cost, recovery architecture, routing parameters, candidate pools and entity scale. Use `--execute` to run the generated jobs; omit it to export their plans and YAMLs. Pool sweeps use 5/10/15/20/40 candidates. Scale sweeps use M=20 and 5K/10K/25K/50K/100K entities.

```bash
python run.py suite --config configs/experiments/snbc_main.yaml --profile budgets --out runs/budgets --execute
python run.py suite --config configs/experiments/snbc_main.yaml --profile mechanism --out runs/mechanism --execute
python run.py suite --config configs/experiments/snbc_main.yaml --profile ablations --out runs/ablations --execute
python run.py suite --config configs/experiments/snbc_main.yaml --profile recovery --out runs/recovery --execute
python run.py suite --config configs/experiments/snbc_main.yaml --profile sensitivity --out runs/sensitivity --execute
python run.py suite --config configs/experiments/snbc_main.yaml --profile pools --out runs/pools --execute
python run.py suite --config configs/experiments/snbc_main.yaml --profile scale --out runs/scale --execute
```

<!-- section: calibration -->

## Router controls and calibration

The consumer experiment compares the complete router, KL-only training, direct teacher, straight-through, Gumbel Top-r and Sparsemax using common acquired evidence and teacher targets. Full-grid calibration evaluates 243 combinations of teacher and router parameters, selects on calibration data and evaluates the selected configuration on test data.

Use the checkpoint corresponding to each evidence-mask/neural-seed replicate.

```bash
python run.py consumers --checkpoint runs/snbc_main/fame_scm_active/mask_0/seed_0/build_state.pkl --out runs/consumers
python run.py calibrate --checkpoint runs/snbc_main/fame_scm_active/mask_0/seed_0/build_state.pkl --out runs/calibration --full-grid
```

<!-- section: transfer -->

## Transfer and equal-time experiments

Identity holdout excludes complete identity groups from competence-model and router fitting. `--component 1` selects the product identity; `--component 0` selects the terminal/store identity. Equal-time experiments use declared construction deadlines and select the latest completed calibrated checkpoint. Preparation accounting is controlled by `--preparation`.

```bash
python run.py holdout --config configs/experiments/snbc_main.yaml --out runs/unseen_products --component 1 --fraction 0.2
python run.py holdout --config configs/experiments/snbc_main.yaml --out runs/unseen_terminals --component 0 --fraction 0.2
python run.py equal-time --config configs/experiments/snbc_main.yaml --out runs/equal_time --seconds 3600 7200 --preparation included
```

<!-- section: insertion -->

## Incremental expert insertion

Insert CatBoost, PatchTST, N-HiTS, DeepAR and Chronos independently into the fitted expert pool. The update trains the appended expert factor, bias and router head while retaining the existing parameters. Screening compares SCM, random acquisition and SoftImpute at 0.1/0.2/0.4 observation ratios against full-column acquisition.

Run one insertion, or use the checkpoint template to run all five experts across the configured repetitions:

```bash
python run.py insertion --expert catboost --checkpoint runs/snbc_main/fame_scm_active/mask_0/seed_0/build_state.pkl --out runs/insert_catboost --ratios 0.1 0.2 0.4 1.0
python scripts/extended_suite.py insertion --config configs/experiments/snbc_main.yaml --out runs/all_insertions --checkpoint "runs/snbc_main/fame_scm_active/mask_{mask}/seed_{seed}/build_state.pkl" --execute
```

<!-- section: evaluation -->

## Evaluation and result export

The pipeline exports RMSE, MAE, WAPE, sMAPE and MSE, along with routing diagnostics, acquired-cell counts and expert-evaluation work. `scm.rho` controls the unique mining-cell budget, including initialization. Mining, calibration, testing and audit have separate execution records. Shared expert fitting, feature preparation and cost profiling are reported separately.

Paired comparisons use common entity/time-block resamples and 10,000 bootstrap draws. Neural-seed results are averaged within each evidence mask before aggregation across masks. Hardware and software details are saved in `run_manifest.json`; GPU resource-hours use external telemetry.

```bash
python run.py compare --out runs/snbc_main --method fame_scm_active --reference nn_ei_c --metric rmse --replicates 10000 --block-days 28
python run.py tables --root runs/snbc_main --out runs/snbc_tables
```

<!-- section: outputs -->

### Output files

| File or directory | Contents |
| --- | --- |
| `resolved_config.yaml` | Resolved experiment configuration |
| `run_manifest.json` | Software, hardware and configuration metadata |
| `results.csv`, `aggregate.csv` | Per-run results and aggregated statistics |
| `<method>/mask_<id>/seed_<id>/predictions.csv` | Forecasts, targets and validity masks |
| `<method>/mask_<id>/seed_<id>/execution_ledger.csv` | Executed requests and cached accesses |
| `<method>/mask_<id>/seed_<id>/build_state.pkl` | Research checkpoint for controls and insertion |
| `<method>/mask_<id>/seed_<id>/deployment.pkl` | Deployment model bundle |

Use a separate output directory for each experiment.

<!-- section: deployment -->

## Deployment and inventory replay

Load a deployment bundle to forecast from the latest available history. Set `--device` for the deployment environment. Inventory replay uses a prepared forecast CSV with `entity_id`, `origin`, `target_date` and `prediction`, together with daily inventory state and the configured replenishment policy. Provide consecutive forecast horizons at the issue dates required by the policy. See [data formats](docs/data_formats.md) for replay inputs. Load checkpoints from trusted sources.

```bash
python run.py predict --checkpoint runs/snbc_main/fame_scm_active/mask_0/seed_0/deployment.pkl --data data/snbc/daily_sales_data.csv --out runs/future_predictions.csv --device cuda:0
python run.py inventory --predictions data/inventory_forecasts.csv --daily data/inventory_daily.csv --policy configs/experiments/inventory_policy.yaml --out runs/inventory_replay
```

<!-- section: checks -->

## Project checks

Run the source, configuration and documentation checks. The GPU validator checks device execution, core model placement and deployment prediction.

```bash
python scripts/test.py
python scripts/check_repository.py
python scripts/check_docs.py
python scripts/validate_gpu.py --device cuda:0 --out runs/verify_gpu
```

<!-- section: repository -->

## Repository guide

| Path | Purpose |
| --- | --- |
| `fame_scm/` | Method implementation and public API |
| `configs/datasets/` | Dataset schemas and temporal splits |
| `configs/experts/` | Candidate expert pools |
| `configs/experiments/` | Main, smoke and inventory configurations |
| `scripts/` | Dataset runners, experiment suites and checks |
| `tests/` | Regression and contract tests |
| `docs/` | Methodology, protocols and output formats |
| `metadata/` | Paper references and experiment correspondence |



<!-- section: citation -->

## Citation and license

Please cite **Sparse Competence Mining for Forecastability-Aware Expert Routing in Heterogeneous Time Series** when using this repository. Author and citation metadata are provided in [CITATION.cff](CITATION.cff).


