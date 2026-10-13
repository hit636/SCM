"""Check pinned FFORMPP/JAX runtime without importing Torch or CUDA models."""

import importlib.metadata, json, sys

report = {"python": sys.version, "executable": sys.executable}
for package in ["jax", "jaxlib", "ml-dtypes", "numpy", "movingknots"]:
    try:
        report[package] = importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        report[package] = "not installed"
try:
    import nvidia.cuda_nvcc as nvcc

    report["cuda_nvcc_file"] = nvcc.__file__
    report["cuda_nvcc_paths"] = list(nvcc.__path__)
except ImportError:
    report["cuda_nvcc_file"] = "not installed"
print(json.dumps(report, indent=2))
if report["jax"] != "0.4.34" or report["jaxlib"] != "0.4.34":
    raise SystemExit(
        "Use this interpreter to install requirements.txt (JAX/JAXLIB both 0.4.34)."
    )
import jax
from movingknots import fformpp

print("JAX CPU devices:", jax.devices("cpu"))
print("FFORMPP import passed.")
