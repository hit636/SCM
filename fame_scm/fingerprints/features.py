"""Training-fitted forecastability features with timestamp-preserving masks."""

import numpy as np
from fame_scm.datasets.temporal import time_step
import pandas as pd
from statsmodels.tsa.seasonal import STL
from fame_scm.datasets.snbc import CATEGORICAL, NUMERIC


def temporal_features(y, valid, season=7, steps_per_day=1, fill_value=None):
    y = np.asarray(y, float)
    valid = np.asarray(valid, bool)
    # Lifecycle starts at the first legally observed/active day, not global grid start.
    first = int(np.flatnonzero(valid)[0]) if valid.any() else len(y)
    y = y[first:]
    valid = valid[first:]
    x = y[valid]
    if not x.size:
        x = np.array([0.0])
    positive = x[x > 0]
    mean = x.mean()
    std = x.std()
    eps = 1e-8
    posmean = positive.mean() if len(positive) else 0.0
    # Filling keeps timestamps intact for spectral/STL features.
    filled = np.where(valid, y, mean if fill_value is None else fill_value)
    if not len(filled):
        filled = np.zeros(season * 2)
    adjacent = valid[1:] & valid[:-1]
    dif = np.abs(np.diff(y)[adjacent])
    dif = dif if len(dif) else np.array([0.0])
    lagcorr = []
    for lag in [1, season]:
        a = filled[:-lag]
        b = filled[lag:]
        lagcorr.append(
            float(np.corrcoef(a, b)[0, 1])
            if len(a) > 2 and a.std() > eps and b.std() > eps
            else 0.0
        )
    freqs = np.abs(np.fft.fftfreq(len(filled)))
    power = np.abs(np.fft.fft(filled)) ** 2
    q = power / (power.sum() + eps)
    month = 30 * steps_per_day
    week = 7 * steps_per_day
    bands = [
        q[freqs <= 1 / month].sum(),
        q[(freqs > 1 / month) & (freqs <= 1 / week)].sum(),
        q[freqs > 1 / week].sum(),
    ]
    trend_strength = season_strength = 0.0
    if len(filled) >= 2 * season and np.var(filled) > eps:
        stl = STL(filled, period=season, robust=True).fit()
        vr = np.var(stl.resid)
        trend_strength = max(0.0, 1 - vr / (np.var(stl.trend + stl.resid) + eps))
        season_strength = max(0.0, 1 - vr / (np.var(stl.seasonal + stl.resid) + eps))
    winsorized = np.clip(filled, *np.quantile(filled, [0.01, 0.99]))
    slope = (
        np.polyfit(np.arange(len(filled)), winsorized, 1)[0] if len(filled) > 1 else 0.0
    )
    return np.array(
        [
            len(y),
            valid.sum(),
            1 - valid.mean() if len(valid) else 1.0,
            np.mean(x == 0),
            len(x) / (len(positive) + eps),
            (positive.std() / (posmean + eps)) ** 2 if len(positive) else 0.0,
            mean,
            std,
            std / (abs(mean) + eps),
            np.quantile(x, 0.5),
            np.quantile(x, 0.9),
            x.max(),
            (dif.std() - dif.mean()) / (dif.std() + dif.mean() + eps),
            np.mean(abs(x - mean) > 3 * std) if std > eps else 0.0,
            slope,
            trend_strength,
            season_strength,
            *lagcorr,
            -np.sum(q * np.log(q + eps)),
            *bands,
            np.mean(filled[-7:]),
            np.mean(filled[-28:]),
            np.var(filled[-7:]),
        ],
        np.float32,
    )


class Forecastability:
    def __init__(self, cfg):
        self.cfg = cfg
        self.vocab = {}
        self.mean = None
        self.std = None
        self.training_fill = {}
        self.global_fill = None

    def raw(self, panel, i, origin):
        season = self.cfg["forecast"]["seasonality"]
        # Use all legally available history; feature scaling fit only on train.
        history = panel.y[i, : origin + 1]
        # Fingerprints describe original signals, not artificial z-score signs.
        transform = panel.audit.get("scalers", {}).get(panel.ids[i])
        if transform is not None:
            history = history * transform["scale"] + transform["mean"]
        fill = self.training_fill.get(panel.ids[i], self.global_fill)
        base = temporal_features(
            history,
            panel.valid[i, : origin + 1],
            season,
            steps_per_day=max(1, round(pd.Timedelta(days=1) / time_step(self.cfg))),
            fill_value=fill,
        )
        row = panel.frames[i].iloc[origin]
        numeric = []
        categorical = []
        for c in self.cfg["data"].get("numeric_features", NUMERIC):
            try:
                v = float(row[c])
            except (ValueError, TypeError, KeyError):
                v = np.nan
            numeric.extend([v if np.isfinite(v) else 0.0, float(not np.isfinite(v))])
        # Historical context response uses only dates <= origin and valid sales.
        context = []
        for c in self.cfg["data"].get("numeric_features", []):
            values = np.asarray(panel.frames[i][c].iloc[: origin + 1], dtype=float)
            if len(values) == 1:
                values = np.repeat(values, origin + 1)
            yy = panel.y[i, : origin + 1]
            legal = panel.valid[i, : origin + 1] & np.isfinite(values)
            v = values[legal]
            q = yy[legal]
            corr = (
                float(np.corrcoef(v, q)[0, 1])
                if len(v) > 2 and v.std() > 1e-8 and q.std() > 1e-8
                else 0.0
            )
            median = float(np.median(v)) if len(v) else 0.0
            high = v > median
            contrast = (
                float(q[high].mean() - q[~high].mean())
                if high.any() and (~high).any()
                else 0.0
            )
            context.extend(
                [
                    float(v.mean()) if len(v) else 0.0,
                    float(v.std()) if len(v) else 0.0,
                    1 - float(legal.mean()),
                    corr,
                    contrast,
                ]
            )
        for c in self.cfg["data"].get("categorical_features", CATEGORICAL):
            value = str(row[c])
            vocab = self.vocab[c]
            categorical.extend([float(value == v) for v in vocab])
            categorical.append(float(value not in vocab))
        date = panel.dates[origin]
        calendar = [
            np.sin(2 * np.pi * date.dayofweek / 7),
            np.cos(2 * np.pi * date.dayofweek / 7),
        ]
        if time_step(self.cfg) < pd.Timedelta(days=1):
            hour = date.hour + date.minute / 60 + date.second / 3600
            calendar.extend(
                [np.sin(2 * np.pi * hour / 24), np.cos(2 * np.pi * hour / 24)]
            )
        observed = np.flatnonzero(panel.valid[i, : origin + 1])
        recency = origin - observed[-1] if len(observed) else origin + 1
        return np.r_[
            base, numeric, context, categorical, calendar, recency, base[23] - base[24]
        ].astype(np.float32)

    def fit(self, panel):
        end = panel.stages["train"].end
        self.training_fill = {}
        for i, entity in enumerate(panel.ids):
            history = panel.y[i, : end + 1][panel.valid[i, : end + 1]]
            transform = panel.audit.get("scalers", {}).get(entity)
            if transform is not None:
                history = history * transform["scale"] + transform["mean"]
            self.training_fill[entity] = float(history.mean()) if len(history) else 0.0
        self.global_fill = float(np.mean(list(self.training_fill.values())))
        self._cache = {}
        for c in self.cfg["data"].get("categorical_features", CATEGORICAL):
            values = set()
            for frame in panel.frames:
                values.update(
                    str(x) for x in frame[c].iloc[: end + 1].dropna().unique()
                )
            self.vocab[c] = sorted(values)
        x = np.stack([self.raw(panel, i, end) for i in range(len(panel.ids))])
        self.mean = x.mean(0)
        self.std = np.maximum(x.std(0), 1.0)
        return self

    def __getstate__(self):
        state = self.__dict__.copy()
        state.pop("_cache", None)
        return state

    def transform(self, panel, origins):
        if isinstance(origins, (int, np.integer)):
            origins = [origins] * len(panel.ids)
        key = (id(panel), tuple(origins))
        if not hasattr(self, "_cache"):
            self._cache = {}
        if key in self._cache:
            return self._cache[key]
        result = np.stack(
            [
                (self.raw(panel, i, o) - self.mean) / self.std
                for i, o in enumerate(origins)
            ]
        ).astype(np.float32)
        if self.cfg.get("features", {}).get("disabled", False):
            result = np.zeros_like(result)
        self._cache[key] = result
        return result

    def state(self):
        return {
            "vocab": self.vocab,
            "mean": self.mean.tolist(),
            "std": self.std.tolist(),
            "version": "origin_safe",
            "training_fill": self.training_fill,
            "global_fill": self.global_fill,
        }


def training_fallback_scales(panel, features, cfg):
    end = panel.stages["train"].end
    floor = cfg["scm"]["scale_floor"]
    scales = []
    strata = []
    for i in range(len(panel.ids)):
        y = panel.y[i, : end + 1]
        mask = panel.valid[i, : end + 1]
        x = y[mask]
        diff = np.diff(y)[mask[1:] & mask[:-1]]
        scale = max(float(np.std(diff)) if len(diff) else float(np.std(x)), floor)
        scales.append(scale)
        strata.append((int(mask.sum() >= 90), int(np.mean(x == 0) > 0.5)))
    global_scale = max(float(np.median(scales)), floor)
    lookup = {
        str(s): float(np.median([scales[j] for j, v in enumerate(strata) if v == s]))
        for s in set(strata)
    }
    fallback = np.array([max(lookup.get(str(s), global_scale), floor) for s in strata])
    return fallback, {
        "definition": "training history >=90 observed days x zero ratio >0.5",
        "scale_proxy": "training consecutive-valid first-difference SD, stratum median",
        "stratum_scales": lookup,
        "global_scale": global_scale,
        "positive_floor": floor,
    }
