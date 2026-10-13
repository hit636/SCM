"""Public Python API usable directly from an unpacked source checkout."""

from copy import deepcopy
from pathlib import Path

from fame_scm.settings import load_settings

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def load_config(path=None):
    """Load composed YAML; relative config/data/output paths use the project root."""
    path = (
        Path(path) if path is not None else Path("configs/experiments/quick_demo.yaml")
    )
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    config = load_settings(path)
    for section, key in [("data", "path"), ("output", None)]:
        value = config[section][key] if key else config[section]
        resolved = Path(value).expanduser()
        if not resolved.is_absolute():
            resolved = PROJECT_ROOT / resolved
        if key:
            config[section][key] = str(resolved)
        else:
            config[section] = str(resolved)
    return config


def make_demo(path, *, series=32, days=420, seed=42):
    """Write synthetic daily data; this is not a real benchmark data generator."""
    from fame_scm.datasets.snbc import make_demo as generate

    return generate(
        Path(path).expanduser().resolve(), series=series, days=days, seed=seed
    )


def train(config=None, *, data=None, out=None, methods=None, device=None):
    """Fit, calibrate and evaluate; return the measured per-run pandas DataFrame.

    Pass a composed config dictionary or YAML path. The dictionary is copied.
    Explicit data/out overrides resolve relative to the caller's current directory.
    Existing completed outputs are never overwritten.
    """
    from fame_scm.orchestration.experiment import run_experiment

    config = deepcopy(config) if isinstance(config, dict) else load_config(config)
    if device is not None:
        config["device"] = str(device)
    if data is not None:
        config["data"]["path"] = str(Path(data).expanduser().resolve())
    if out is not None:
        config["output"] = str(Path(out).expanduser().resolve())
    return run_experiment(config, methods=methods)


def audit(out):
    """Recompute measured predictions, budgets and evidence-ledger invariants."""
    from fame_scm.evaluation.audit import audit_directory

    return audit_directory(Path(out))


def compare(out, method, reference, **options):
    """Compute paired bootstrap statistics from compatible prediction records."""
    from fame_scm.evaluation.statistics import paired_bootstrap

    return paired_bootstrap(out, method, reference, **options)


def predict(checkpoint, data, out, *, device=None):
    """Forecast from a trusted local deployment checkpoint; return output path."""
    from fame_scm.routing.inference import predict_file

    return predict_file(checkpoint, data, out, device=device)


def tables(root, out):
    """Export measured summaries and evidence accounts without paper targets."""
    from fame_scm.evaluation.tables import export_tables

    return export_tables(root, out)
