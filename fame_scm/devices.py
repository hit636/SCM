"""Explicit CPU/CUDA selection without silent fallback."""

import os
import re
import torch


def resolve_device(value):
    value = str(value)
    if value == "cpu":
        return torch.device("cpu")
    if not re.fullmatch(r"cuda(?::[0-9]+)?", value):
        raise ValueError("device must be cpu, cuda or cuda:<visible GPU index>")
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested but is unavailable. Install a CUDA-enabled Torch "
            "wheel and check the NVIDIA driver/GPU visibility, or explicitly "
            "select --device cpu for a CPU run. No CPU fallback was performed."
        )
    index = int(value.split(":")[1]) if ":" in value else 0
    if index >= torch.cuda.device_count():
        raise ValueError(f"CUDA device index {index} is not visible")
    return torch.device(f"cuda:{index}")


def prepare_device(value):
    # Must precede CUDA linear algebra initialization for deterministic training.
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    device = resolve_device(value)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    return device


def device_report(value, *, probe=False):
    device = prepare_device(value)
    report = {
        "requested": str(value),
        "selected": str(device),
        "torch_version": torch.__version__,
        "torch_cuda_build": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "visible_gpu_count": torch.cuda.device_count(),
        "gpu_name": None,
        "gpu_total_memory_bytes": None,
        "probe": "not_requested",
    }
    if device.type == "cuda":
        properties = torch.cuda.get_device_properties(device)
        report.update(
            gpu_name=properties.name, gpu_total_memory_bytes=properties.total_memory
        )
    if probe:
        a = torch.arange(16, dtype=torch.float32, device=device).reshape(4, 4)
        product = a @ a.T
        if not bool(torch.isfinite(product).all()):
            raise RuntimeError("Device matrix multiplication returned nonfinite values")
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        report["probe"] = "passed"
    return report
