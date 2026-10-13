"""Frequency helpers shared by training, evaluation and deployment."""

import pandas as pd


def time_step(cfg):
    offset = pd.tseries.frequencies.to_offset(cfg["data"].get("frequency", "D"))
    try:
        step = pd.Timedelta(offset.nanos, unit="ns")
    except ValueError as exc:
        raise ValueError("Use a fixed sampling frequency, e.g. D, h, 15min") from exc
    if step <= pd.Timedelta(0):
        raise ValueError("Sampling frequency must be positive")
    return step


def timestamp_text(value, cfg):
    value = pd.Timestamp(value)
    return (
        str(value.date())
        if time_step(cfg) == pd.Timedelta(days=1)
        else value.isoformat()
    )
