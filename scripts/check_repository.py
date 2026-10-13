"""Validate repository structure, configuration composition, and local links."""

import ast
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from fame_scm.settings import load_settings


def check_repository():
    requirements = sorted(ROOT.glob("requirements*.txt"))
    if requirements != [ROOT / "requirements.txt"]:
        raise ValueError("Keep only one dependency file: requirements.txt")
    python_files = [
        p
        for group in ["fame_scm", "scripts", "tests"]
        for p in (ROOT / group).rglob("*.py")
    ] + [ROOT / "run.py", ROOT / "example_api.py"]
    for path in python_files:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.ImportFrom)
                and node.module
                and node.module.startswith("fame_scm.")
            ):
                module = ROOT / Path(*node.module.split("."))
                if (
                    not module.with_suffix(".py").exists()
                    and not (module / "__init__.py").exists()
                ):
                    raise ValueError(f"Missing local module: {node.module} in {path}")
    configs = sorted((ROOT / "configs").rglob("*.yaml"))
    for path in configs:
        if path.name != "inventory_policy.yaml":
            load_settings(path)
    for dataset, horizon in [
        ("snbc", 14),
        ("m5", 28),
        ("favorita", 28),
        ("electricity", 168),
        ("traffic", 168),
    ]:
        config = load_settings(ROOT / f"configs/experiments/{dataset}_main.yaml")
        if config["forecast"]["horizon"] != horizon:
            raise ValueError(f"Incorrect main horizon: {dataset}")
        if (
            len(config["evaluation"]["mask_seeds"]) != 5
            or len(config["evaluation"]["neural_seeds"]) != 3
        ):
            raise ValueError(f"Incorrect main repetitions: {dataset}")
        if config["scm"]["members"] != 5 or config["scm"]["batches"] != 4:
            raise ValueError(f"Incorrect main ensemble/acquisition settings: {dataset}")
    markdown_files = (
        [
            ROOT / "README.md",
            ROOT / "THIRD_PARTY.md",
            ROOT / "CONTRIBUTING.md",
            ROOT / "使用说明.md",
        ]
        + list((ROOT / "docs").glob("*.md"))
        + [ROOT / "data/README.md"]
    )
    for path in markdown_files:
        source = path.read_text(encoding="utf-8")
        for link in re.findall(r"\]\(([^)]+)\)", source):
            if link.startswith(("https://", "http://", "#")):
                continue
            target = path.parent / link.split("#", 1)[0]
            if not target.exists():
                raise ValueError(f"Broken documentation link: {path}: {link}")
    return {
        "python_files": len(python_files),
        "yaml_configs": len(configs),
        "markdown_files": len(markdown_files),
        "dependency_files": len(requirements),
        "status": "passed",
    }


if __name__ == "__main__":
    print(json.dumps(check_repository(), indent=2))
