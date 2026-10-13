# Executed validation — version 2.3

Five main experiment configs now require CUDA; CPU smoke configs remain available.
The implementation supports explicit indexed device selection and deployment on
CPU/CUDA. The development host has Torch 2.6.0+cpu and no visible GPU, so actual
successful CUDA execution and GPU performance remain unverified.

| Current check | Result |
| --- | --- |
| Unit tests | **90 passed, 1 real-CUDA test skipped** (91 total) |
| CPU device probe and doctor | Actual matrix multiplication passed; dependencies ready |
| Unavailable-CUDA probe and doctor | Explicit failure, exit 1; no CPU fallback |
| GPU validator without hardware | Exit 1 before creating output; no pass report produced |
| English/Chinese README | **14 paired sections, 12 identical Bash blocks** |
| Source API | Two methods and prediction/evidence audit passed on CPU |
| Core manuscript integration on CPU | Eleven methods, six consumers, two calibration fits, strict protocol and 10K paired bootstrap draws passed |
| CPU prediction equivalence | All eleven prediction CSVs byte-identical to version 2.1 under the same synthetic protocol |
| Extended CPU backends | All 40 candidates finite and save/reload checks passed |
| Extended workflow | Seven insertion runs, two deadline runs, deployment and inventory replay passed on CPU |
| Repository, formatting and builds | Passed; one dependency file retained |

The GPU unit test and [real CUDA validator](../scripts/validate_gpu.py) are provided
for execution on an actual GPU host. Simulated availability/index checks only test
failure and configuration boundaries; they are not successful CUDA execution.
Extended-backend GPU behavior, GPU speed/memory figures, distributed two-GPU
training and original paper benchmark scores are not verified here.

Current evidence is in [verification](verification) and [validation.json](validation.json).
The nested record retains 2.2/2.1/2.0 validation. Historical logs may reference the
older layouts or dependency filenames. CI uses CPU on hosted runners; Python
3.10/3.11/3.12 is configured, but only Linux/Python 3.12 was executed here.

## Repeat CPU regression checks

```bash
python -m pip install torch==2.6.0+cpu --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
python scripts/check_device.py --device cpu
python scripts/test.py
python scripts/check_repository.py
python scripts/check_docs.py
bash scripts/run_api.sh --out runs/api_example --device cpu
python scripts/validate_revision.py --out runs/verify_revision
python scripts/validate_extended.py --out runs/verify_extended --all-experts
python -m black --check fame_scm scripts tests run.py example_api.py
python -m build
```

Use fresh output directories and the frozen Chronos weights for the all-backend
check. See [GPU instructions](gpu.md) for the alternative CUDA installation and
real-device validation. Main experiments now require a GPU or an explicit CPU
override. One `requirements.txt` and direct local imports remain the default.
GPU hours remain null without external telemetry. These synthetic checks validate
workflow contracts, not numerical reproduction of the original paper.
