"""Executable SNBC expert pool: genuine statistical, tree and upstream neural models."""

from pathlib import Path
from types import SimpleNamespace
import importlib, logging, pickle, time, warnings
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.linear_model import Ridge
from lightgbm import LGBMRegressor
from fame_scm.datasets.temporal import time_step
from fame_scm.devices import prepare_device


def tree_rows(history, origin_date, horizon, frequency="D"):
    x = np.asarray(history, float)
    mean = x.mean()
    std = x.std()
    base = np.r_[x, mean, std, np.mean(x[-7:]), np.mean(x[-28:]), np.mean(x == 0)]
    rows = []
    for h in range(1, horizon + 1):
        date = origin_date + h * pd.Timedelta(
            pd.tseries.frequencies.to_offset(frequency).nanos, unit="ns"
        )
        day = date.dayofweek
        row = np.r_[base, h, np.sin(2 * np.pi * day / 7), np.cos(2 * np.pi * day / 7)]
        if (
            pd.tseries.frequencies.to_offset(frequency).nanos
            < pd.Timedelta(days=1).value
        ):
            hour = date.hour + date.minute / 60 + date.second / 3600
            row = np.r_[
                row, np.sin(2 * np.pi * hour / 24), np.cos(2 * np.pi * hour / 24)
            ]
        rows.append(row)
    return np.array(rows, np.float32)


class ExpertPool:
    def __init__(self, cfg, panel):
        self.cfg = cfg
        self.panel = panel
        self.names = list(cfg["experts"]["names"])
        self.models = {}
        self.logs = []
        self.lookback = cfg["forecast"]["lookback"]
        self.h = cfg["forecast"]["horizon"]
        self.device = prepare_device(cfg["device"])
        self.fill = []
        end = panel.stages["train"].end
        for i in range(len(panel.ids)):
            x = panel.y[i, : end + 1][panel.valid[i, : end + 1]]
            self.fill.append(float(np.mean(x)) if len(x) else 0.0)

    def availability(self, origins):
        n = len(self.panel.ids)
        a = np.ones((n, len(self.names)), bool)
        limits = self.cfg["experts"].get("min_history", {})
        required = self.cfg["experts"].get("required_covariates", {})
        for i in range(n):
            valid = self.panel.valid[i]
            seen = np.flatnonzero(valid)
            first = int(seen[0]) if len(seen) else len(valid)
            for j, name in enumerate(self.names):
                minimum = limits.get(name, self.lookback)
                for o in origins:
                    a[i, j] &= o - first + 1 >= minimum and bool(
                        valid[max(0, o - self.lookback + 1) : o + 1].any()
                    )
                    for c in required.get(name, []):
                        try:
                            a[i, j] &= np.isfinite(
                                float(self.panel.frames[i].iloc[o][c])
                            )
                        except (KeyError, TypeError, ValueError):
                            a[i, j] = False
        return a

    def windows(self):
        # Reservoir sample lightweight indices BEFORE allocating window tensors.
        end = self.panel.stages["train"].end
        cap = self.cfg["experts"]["max_train_windows"]
        rng = np.random.default_rng(self.cfg["seed"])
        pairs = []
        seen = 0
        for i in range(len(self.panel.ids)):
            for o in range(
                self.lookback - 1,
                end - self.h + 1,
                self.cfg["forecast"]["train_stride"],
            ):
                mask = self.panel.valid[i, o + 1 : o + self.h + 1]
                if mask.mean() < self.cfg["forecast"]["min_valid_target_fraction"]:
                    continue
                if not self.panel.valid[i, max(0, o - self.lookback + 1) : o + 1].any():
                    continue
                seen += 1
                if not cap or len(pairs) < cap:
                    pairs.append((i, o))
                else:
                    j = int(rng.integers(seen))
                    if j < cap:
                        pairs[j] = (i, o)
        if not pairs:
            raise ValueError("No full-horizon training windows")
        pairs.sort()
        self.window_entities = np.array([i for i, o in pairs])
        self.eligible_windows = seen
        xs = [self.panel.history(i, o, self.lookback, self.fill[i]) for i, o in pairs]
        ys = [self.panel.y[i, o + 1 : o + self.h + 1] for i, o in pairs]
        valid = [self.panel.valid[i, o + 1 : o + self.h + 1] for i, o in pairs]
        return (
            np.array(xs, np.float32),
            np.array(ys, np.float32),
            np.array(valid, bool),
            np.array([o for i, o in pairs]),
        )

    def context_rows(self, i, origin):
        """Origin-known context, repeated over horizon; no future realized covariates."""
        cols = self.cfg["data"].get("numeric_features", [])
        row = self.panel.frames[i].iloc[origin]
        values = []
        for c in cols:
            try:
                v = float(row[c])
            except (KeyError, TypeError, ValueError):
                v = np.nan
            values.extend([v if np.isfinite(v) else 0.0, float(not np.isfinite(v))])
        return np.tile(np.asarray(values, np.float32), (self.h, 1))

    def deployment_copy(self):
        from copy import copy, deepcopy

        result = copy(self)
        result.training_ids = list(self.panel.ids)
        result.panel = None
        result.models = deepcopy(self.models)
        return result.to("cpu")

    def to(self, device):
        from copy import deepcopy

        self.device = prepare_device(device)
        self.cfg = deepcopy(self.cfg)
        self.cfg["device"] = str(self.device)
        for model in self.models.values():
            if isinstance(model, torch.nn.Module):
                model.to(self.device).eval()
            elif hasattr(model, "to"):
                model.to(self.device)
        return self

    def fit(self):
        x, y, valid, origins = self.windows()
        self.train_windows = len(x)
        ec = self.cfg["experts"]
        seed = self.cfg["seed"]
        tree_x = None
        tree_y = None
        for name in self.names:
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
            started = time.perf_counter()
            if (
                name
                in [
                    "catboost",
                    "nbeats",
                    "nhits",
                    "patchtst",
                    "deepar",
                    "tcn",
                    "gru",
                    "tft",
                    "mamba",
                    "chronos",
                ]
                or "__variant" in name
            ):
                from fame_scm.experts.extended_experts import ExtendedExpert

                self.models[name] = ExtendedExpert(name, self.cfg).fit(
                    self, x, y, valid, origins
                )
            elif name in ["linear", "lightgbm", "xgboost"]:
                if tree_x is None:
                    tree_x = np.concatenate(
                        [
                            np.c_[
                                tree_rows(
                                    a,
                                    self.panel.dates[o],
                                    self.h,
                                    self.cfg["data"].get("frequency", "D"),
                                ),
                                self.context_rows(i, o),
                            ]
                            for a, o, i in zip(x, origins, self.window_entities)
                        ]
                    )[valid.ravel()]
                    tree_y = y.ravel()[valid.ravel()]
                if name == "linear":
                    model = Ridge(alpha=ec.get("ridge_alpha", 1.0))
                elif name == "lightgbm":
                    model = LGBMRegressor(
                        n_estimators=ec["tree_estimators"],
                        num_leaves=ec.get("num_leaves", 15),
                        min_child_samples=10,
                        verbosity=-1,
                        n_jobs=self.cfg["threads"],
                        random_state=seed,
                    )
                else:
                    from xgboost import XGBRegressor

                    model = XGBRegressor(
                        n_estimators=ec["tree_estimators"],
                        max_depth=ec.get("tree_depth", 4),
                        n_jobs=self.cfg["threads"],
                        random_state=seed,
                        tree_method="hist",
                    )
                model.fit(tree_x, tree_y)
                self.models[name] = model
            elif name in ["dlinear", "timemixer", "timesnet"]:
                torch.manual_seed(seed)
                module = importlib.import_module(
                    "fame_scm.vendor.tslib.models."
                    + {
                        "dlinear": "DLinear",
                        "timemixer": "TimeMixer",
                        "timesnet": "TimesNet",
                    }[name]
                )
                conf = SimpleNamespace(
                    task_name="long_term_forecast",
                    seq_len=self.lookback,
                    label_len=0,
                    pred_len=self.h,
                    enc_in=1,
                    c_out=1,
                    d_model=ec["d_model"],
                    d_ff=ec["d_ff"],
                    e_layers=ec["e_layers"],
                    dropout=0.1,
                    embed="timeF",
                    freq=("h" if time_step(self.cfg) < pd.Timedelta(days=1) else "d"),
                    moving_avg=ec.get("moving_avg", 7),
                    top_k=2,
                    num_kernels=3,
                    channel_independence=1,
                    down_sampling_layers=1,
                    down_sampling_window=2,
                    down_sampling_method="avg",
                    decomp_method="moving_avg",
                    use_norm=1,
                    num_class=1,
                )
                model = module.Model(conf).to(self.device)
                opt = torch.optim.Adam(model.parameters(), lr=ec["learning_rate"])
                scale = np.maximum(np.mean(abs(x), axis=1, keepdims=True), 1.0)
                ds = TensorDataset(
                    torch.tensor(x / scale)[:, :, None],
                    torch.tensor(y / scale),
                    torch.tensor(valid),
                )
                generator = torch.Generator().manual_seed(seed)
                dl = DataLoader(
                    ds, batch_size=ec["batch_size"], shuffle=True, generator=generator
                )
                for epoch in range(ec["neural_epochs"]):
                    model.train()
                    loss_sum = 0.0
                    count = 0
                    for a, b, mask in dl:
                        a, b, mask = (
                            a.to(self.device),
                            b.to(self.device),
                            mask.to(self.device),
                        )
                        opt.zero_grad()
                        pred = model(a, None, None, None)[:, -self.h :, 0]
                        loss = (pred[mask] - b[mask]).square().mean()
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                        opt.step()
                        loss_sum += float(loss.detach()) * len(a)
                        count += len(a)
                    self.logs.append(
                        {
                            "expert": name,
                            "epoch": epoch,
                            "training_mse_scaled": loss_sum / count,
                        }
                    )
                model.eval()
                self.models[name] = model
            elif name not in ["sarima", "ets", "prophet", "tsb"]:
                raise ValueError(f"Unknown implemented expert: {name}")
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
            self.logs.append(
                {
                    "expert": name,
                    "fit_seconds": time.perf_counter() - started,
                    "training_windows": len(x),
                }
            )
        return self

    def predict(self, name, i, origin):
        p = self.panel
        history = p.history(i, origin, self.lookback, self.fill[i])
        season = self.cfg["forecast"]["seasonality"]
        if (
            name
            in [
                "catboost",
                "nbeats",
                "nhits",
                "patchtst",
                "deepar",
                "tcn",
                "gru",
                "tft",
                "mamba",
                "chronos",
            ]
            or "__variant" in name
        ):
            pred = self.models[name].predict(self, i, origin, history)
        elif name in ["linear", "lightgbm", "xgboost"]:
            model = self.models[name]
            x = np.c_[
                tree_rows(
                    history,
                    p.dates[origin],
                    self.h,
                    self.cfg["data"].get("frequency", "D"),
                ),
                self.context_rows(i, origin),
            ]
            pred = model.booster_.predict(x) if name == "lightgbm" else model.predict(x)
        elif name in ["dlinear", "timemixer", "timesnet"]:
            scale = max(float(np.mean(abs(history))), 1.0)
            with torch.no_grad():
                pred = (
                    self.models[name](
                        torch.tensor(history / scale, device=self.device)[
                            None, :, None
                        ],
                        None,
                        None,
                        None,
                    )[0, -self.h :, 0]
                    .cpu()
                    .numpy()
                    * scale
                )
        elif name == "tsb":
            probability = float(np.mean(history > 0))
            positive = history[history > 0]
            size = float(positive.mean()) if len(positive) else 0.0
            alpha = self.cfg["experts"].get("alpha", 0.1)
            for value in history:
                probability = alpha * (value > 0) + (1 - alpha) * probability
                if value > 0:
                    size = alpha * value + (1 - alpha) * size
            pred = np.full(self.h, probability * size)
        elif name == "ets":
            from statsmodels.tsa.holtwinters import ExponentialSmoothing

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                fit = ExponentialSmoothing(
                    history,
                    trend="add",
                    damped_trend=self.cfg["experts"].get("damped_trend", False),
                    seasonal="add",
                    seasonal_periods=season,
                    initialization_method="estimated",
                ).fit(optimized=True)
                pred = fit.forecast(self.h)
        elif name == "sarima":
            from statsmodels.tsa.statespace.sarimax import SARIMAX

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                fit = SARIMAX(
                    history,
                    order=tuple(self.cfg["experts"].get("order", [1, 0, 0])),
                    seasonal_order=(1, 0, 0, season),
                    trend="c",
                    enforce_stationarity=False,
                    enforce_invertibility=False,
                ).fit(disp=False, maxiter=self.cfg["experts"]["sarima_maxiter"])
                pred = fit.forecast(self.h)
        elif name == "prophet":
            from prophet import Prophet

            logging.getLogger("cmdstanpy").setLevel(logging.ERROR)
            dates = p.dates[origin - self.lookback + 1 : origin + 1]
            model = Prophet(
                yearly_seasonality=False,
                weekly_seasonality=True,
                daily_seasonality=(time_step(self.cfg) < pd.Timedelta(days=1)),
                uncertainty_samples=0,
                changepoint_prior_scale=self.cfg["experts"].get(
                    "changepoint_prior_scale", 0.05
                ),
            )
            model.fit(pd.DataFrame({"ds": dates, "y": history}))
            future = pd.DataFrame(
                {
                    "ds": pd.date_range(
                        p.dates[origin] + time_step(self.cfg),
                        periods=self.h,
                        freq=time_step(self.cfg),
                    )
                }
            )
            pred = model.predict(future)["yhat"].to_numpy()
        else:
            raise ValueError(name)
        pred = np.asarray(pred, float).reshape(-1)
        if pred.shape != (self.h,) or not np.isfinite(pred).all():
            raise ValueError(
                f"{name} returned invalid forecasts for entity {i}, origin {origin}"
            )
        return (
            np.maximum(pred, 0.0)
            if self.cfg["forecast"].get("clip_nonnegative", True)
            else pred
        )

    def save(self, directory):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        for name, model in self.models.items():
            if isinstance(model, torch.nn.Module):
                torch.save(model.state_dict(), directory / f"{name}.pt")
            else:
                with (directory / f"{name}.pkl").open("wb") as f:
                    pickle.dump(model, f)
        pd.DataFrame(self.logs).to_csv(directory / "expert_training.csv", index=False)
