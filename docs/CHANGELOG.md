# Changelog

## 2.3.0 — explicit CUDA experiments and deployment

- Default all five main experiments to CUDA; retain CPU smoke configurations.
- Add indexed device overrides, strict CUDA preflight and actual device metadata.
- Restore selected-device inference from portable CPU deployment bundles.
- Move IMC and GP latent fitting tensors to the selected device.
- Preserve GPU indices in NeuralForecast trainers and device-specific timing.
- Bind frozen cost profiles to the device, Torch build and GPU name.
- Add a real-hardware GPU validator; retain explicit unverified CUDA status here.

## 2.2.0 — one dependency file and Bash README examples

- Keep only `requirements.txt`; CI and distribution metadata use the same file.
- Preserve the historical full environment snapshot as inspection-only JSON.
- Encapsulate the Python API example behind `scripts/run_api.sh`.
- Keep both README languages aligned and require all example blocks to be Bash.
- Retain direct local imports without installing the FAME-SCM project.

## 2.1.0 — direct source API and aligned documentation

- Move the local `fame_scm` package to the repository root for direct imports.
- Add `fame_scm.api` and a runnable `example_api.py`; no project installation.
- Make core/dev/full requirements contain third-party libraries only.
- Remove project installation from CI; validate source API and bilingual examples.
- Align every English/Chinese README section and all executable examples.
- Verify 83 unit tests, uninstalled API execution and eleven-method prediction
  equivalence against the 2.0 synthetic protocol.

## 2.0.0 — current-manuscript engineering revision

- Add NN-EI/c, independent neural recovery, no-Huber/no-normalization controls.
- Add common-teacher consumers and calibration-only teacher/router grid search.
- Estimate/freeze ranking margin from acquired seed origin evidence; normalize
  minibatch objectives using global observed entry/pair counts with chunked pairs.
- Add total construction request caps, complete per-stage accounting, audit-only
  measured weighted work and component-loss stability diagnostics.
- Correct zero-SD expected improvement, final IMC prediction, robust trend features,
  resource-monitor failures and repeated insertion checkpoint pairing.
- Add pool/scale/request plans, real SoftImpute new-column screening and measured
  full-column weights, 10K centered paired bootstrap, strict protocol binding,
  preflight imports and measured-only table export.
- Package core/optional dependencies, pin incompatible optional transitive versions,
  record the tested environment, refresh validation and reproducibility documentation.
- Preserve the original license and all vendored third-party notices.
