# GPU execution and verification

Five main experiment configs select `cuda:0`. Smoke configs keep `cpu` so that
small checks and GitHub-hosted CPU CI remain runnable. The device field accepts
`cpu`, `cuda` (visible index zero), or `cuda:N`. Requested CUDA never silently
falls back to CPU. Configuration loading is hardware-independent, so dry-run
plans can be generated on a CPU host; execution and the device doctor validate
real availability.

## Install and check

Use an NVIDIA GPU and a compatible working driver. The pinned Torch 2.6 release
provides CUDA 11.8, 12.4 and 12.6 wheels; choose a build compatible with your
hardware according to the [official instructions](https://pytorch.org/get-started/previous-versions/).
The README uses CUDA 12.4 explicitly. Its wheel suffix replaces a prior CPU build,
while `torch==2.6.0` in the sole dependency file accepts that CUDA wheel.

```bash
python -m pip install torch==2.6.0+cu124 --index-url https://download.pytorch.org/whl/cu124
python -m pip install -r requirements.txt
python scripts/check_device.py --device cuda:0
python run.py doctor --config configs/experiments/snbc_main.yaml --device cuda:0
```

The probe executes an actual matrix multiplication on the selected device.
Missing CUDA returns exit code 1. Doctor checks dependency imports and device
execution separately, including Torch build, CUDA availability, GPU name,
visible GPU count and total memory. A CUDA wheel alone is insufficient when
hardware or driver access is missing.

## Scope and placement

SCM models, the distilled router, independent neural recovery, core neural
experts, Mamba reference, NeuralForecast experts and Chronos use the selected
Torch device. IMC and GP latent recovery fitting also use it; GP conditioning
returns to NumPy/CPU. Statistical experts, trees, feature computation, acquisition
ranking and metrics remain CPU workloads with the supplied settings. The supplied
JAX/JAXlib dependencies keep FFORMPP-B on CPU. Tree GPU variants are not enabled.

NeuralForecast's trainer receives the selected visible CUDA index. Portable
`deployment.pkl` files store CPU weights; prediction restores both the expert pool
(including nested variants) and router to the requested device. Inference defaults
to the checkpoint's training config; use `--device cpu` to deploy a CUDA-trained
bundle on a CPU host. Research build-state pickle files may contain live CUDA
tensors and require compatible GPU hardware when loaded. Only load trusted files.

```bash
python run.py predict --checkpoint runs/snbc_gpu/fame_scm_active/mask_0/seed_0/deployment.pkl --data data/snbc/daily_sales_data.csv --out runs/cpu_predictions.csv --device cpu
bash scripts/run_api.sh --out runs/gpu_api_example --device cuda:0
python scripts/run_datasets.py --profile main --device cuda:0 --out runs/five_datasets_gpu --dry-run
```

Training and execution profiles record the selected device. Per-request timing
synchronizes that device. Frozen expert cost profiles include the selected device,
Torch build and GPU name in their signature; they cannot be reused across CPU/GPU
or changed GPU-model profiles. Separate profiling runs and contention can still
change timings; preserve the exact hardware/execution record for comparisons.
GPU resource hours remain null without external telemetry; GPU availability or
wall time is not a measurement of utilization.

## Real CUDA validation

```bash
python scripts/validate_gpu.py --device cuda:0 --out runs/verify_gpu
```

This script requires CUDA before creating outputs. It executes five core methods,
including IMC and GP latent recovery, checks actual expert/SCM/router parameter
devices, audits predictions and runs both GPU and CPU deployment from one portable
bundle. It writes `gpu_validation.json` only after those checks pass. It does not
validate all extended backends, benchmark scores or GPU-hour accounting. Unit tests
include a real CUDA backward/recovery test that is skipped if no GPU is available.

The development host has Torch 2.6.0+cpu and no visible GPU. CPU regressions and
explicit unavailable-CUDA behavior were executed here; real CUDA training,
extended-backend GPU execution and GPU speed/memory figures remain unverified.

## Multiple GPUs

Each job selects one device. This release does not implement DDP, DataParallel,
sharded ensembles or automatic two-GPU scheduling. Independent jobs can choose
separate GPUs and separate output directories. Indices refer to the visible set:

```bash
CUDA_VISIBLE_DEVICES=1 python run.py run --config configs/experiments/snbc_main.yaml --device cuda:0 --out runs/snbc_physical_gpu1
```

A process exposing only physical GPU 1 sees it as `cuda:0`. Such an independent-job
setup must not be reported as a verified distributed two-GPU paper experiment.
