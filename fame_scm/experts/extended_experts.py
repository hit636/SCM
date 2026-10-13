"""Real additional backends; missing dependencies/weights fail explicitly."""

from copy import deepcopy
import hashlib
import numpy as np
import pandas as pd
import torch
from torch import nn

EXTRA = {
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
}
NF_NAMES = {
    "nbeats": "NBEATS",
    "nhits": "NHITS",
    "patchtst": "PatchTST",
    "deepar": "DeepAR",
    "tcn": "TCN",
    "gru": "GRU",
    "tft": "TFT",
}


class SelectiveSSM(nn.Module):
    """Mamba-style input-dependent selective SSM, pure Torch reference scan.

    Implements stable diagonal A, input-dependent dt/B/C, causal convolution
    and gated skip. Not a claim of upstream Mamba CUDA kernel equivalence.
    """

    def __init__(self, h, width=16, state_size=8):
        super().__init__()
        self.width = width
        self.state_size = state_size
        self.input = nn.Linear(1, width * 2)
        self.conv = nn.Conv1d(width, width, 3, groups=width, padding=2)
        self.select = nn.Linear(width, width + 2 * state_size)
        self.log_a = nn.Parameter(
            torch.log(torch.arange(1, state_size + 1).float()).repeat(width, 1)
        )
        self.skip = nn.Parameter(torch.ones(width))
        self.norm = nn.LayerNorm(width)
        self.head = nn.Linear(width, h)

    def forward(self, x):
        u, g = self.input(x[..., None]).chunk(2, -1)
        u = torch.nn.functional.silu(
            self.conv(u.transpose(1, 2))[..., : x.shape[1]].transpose(1, 2)
        )
        dt, b, c = torch.split(
            self.select(u), [self.width, self.state_size, self.state_size], -1
        )
        dt = torch.nn.functional.softplus(dt)
        state = u.new_zeros((len(x), self.width, self.state_size))
        outs = []
        for t in range(x.shape[1]):
            step = dt[:, t, :, None]
            a = -self.log_a.exp()[None]
            # Zero-order hold input discretization; all transitions stable.
            trans = torch.exp(step * a)
            drive = torch.expm1(step * a) / a
            state = trans * state + drive * b[:, t, None, :] * u[:, t, :, None]
            outs.append(
                ((state * c[:, t, None, :]).sum(-1) + self.skip * u[:, t])
                * torch.nn.functional.silu(g[:, t])
            )
        return self.head(self.norm(torch.stack(outs, 1)[:, -1]))


class ExtendedExpert:
    def __init__(self, name, cfg):
        self.name = name
        self.cfg = deepcopy(cfg)
        spec = (
            cfg["experts"].get("specs", {}).get(name, {"kind": name, "parameters": {}})
        )
        self.kind = spec["kind"]
        self.params = spec.get("parameters", {})
        self.model = None

    def fit(self, pool, x, y, valid, origins):
        ec = pool.cfg["experts"]
        kind = self.kind
        seed = pool.cfg["seed"]
        self.h = pool.h
        if kind not in EXTRA:
            from fame_scm.experts.forecast_experts import ExpertPool

            c = deepcopy(pool.cfg)
            c["experts"].pop("specs", None)
            c["experts"]["names"] = [kind]
            c["experts"].update(self.params)
            self.model = ExpertPool(c, pool.panel).fit()
            self.model.panel = None
        elif kind == "catboost":
            from catboost import CatBoostRegressor
            from fame_scm.experts.forecast_experts import tree_rows

            a = np.concatenate(
                [
                    np.c_[
                        tree_rows(
                            xx,
                            pool.panel.dates[o],
                            pool.h,
                            pool.cfg["data"].get("frequency", "D"),
                        ),
                        pool.context_rows(i, o),
                    ]
                    for xx, o, i in zip(x, origins, pool.window_entities)
                ]
            )
            self.model = CatBoostRegressor(
                iterations=ec["tree_estimators"],
                random_seed=seed,
                thread_count=pool.cfg["threads"],
                verbose=False,
                allow_writing_files=False,
                **{"depth": 5, **self.params},
            )
            self.model.fit(a[valid.ravel()], y.ravel()[valid.ravel()])
        elif kind in NF_NAMES:
            from neuralforecast import NeuralForecast, models
            from neuralforecast.losses.pytorch import MSE

            kw = dict(
                h=pool.h,
                input_size=pool.lookback,
                max_steps=ec.get("nf_steps", 100),
                random_seed=seed,
                scaler_type="standard",
                batch_size=min(ec["batch_size"], len(x)),
                windows_batch_size=ec["batch_size"],
                learning_rate=ec["learning_rate"],
                val_check_steps=ec.get("nf_steps", 100),
                enable_progress_bar=False,
                logger=False,
                enable_checkpointing=False,
                accelerator="gpu" if pool.device.type == "cuda" else "cpu",
                devices=[pool.device.index] if pool.device.type == "cuda" else 1,
            )
            # Fit independent sampled training windows, preserving valid targets.
            if kind != "deepar":
                kw["loss"] = MSE()
            if kind in ["nbeats", "nhits"]:
                kw["mlp_units"] = [[32, 32]] * 3
            if kind == "patchtst":
                kw.update(
                    hidden_size=16, n_heads=4, encoder_layers=1, patch_len=7, stride=7
                )
            if kind == "deepar":
                kw.update(lstm_hidden_size=32, lstm_n_layers=1, trajectory_samples=20)
            if kind in ["tcn", "gru"]:
                kw.update(encoder_hidden_size=32, decoder_hidden_size=32)
            if kind == "tft":
                kw.update(hidden_size=32, n_head=4)
            kw.update(self.params)
            model = getattr(models, NF_NAMES[kind])(**kw)
            frames = []
            for j, (xx, yy, mask) in enumerate(zip(x, y, valid)):
                a = np.r_[xx, np.where(mask, yy, 0.0)]
                observed = np.r_[np.ones(len(xx)), mask.astype(float)]
                frames.append(
                    pd.DataFrame(
                        {
                            "unique_id": str(j),
                            "ds": pd.date_range(
                                "2000-01-01",
                                periods=len(a),
                                freq=pool.cfg["data"].get("frequency", "D"),
                            ),
                            "y": a,
                            "available_mask": observed,
                        }
                    )
                )
            self.model = NeuralForecast(
                models=[model], freq=pool.cfg["data"].get("frequency", "D")
            )
            self.model.fit(df=pd.concat(frames, ignore_index=True), val_size=0)
            # NF needs weights and fitted scalers, not the training dataset at inference.
        elif kind == "mamba":
            torch.manual_seed(seed)
            self.model = SelectiveSSM(pool.h, **self.params).to(pool.device)
            scale = np.maximum(np.abs(x).mean(1, keepdims=True), 1.0)
            a = torch.tensor(x / scale, device=pool.device)
            b = torch.tensor(y / scale, device=pool.device)
            v = torch.tensor(valid, device=pool.device)
            opt = torch.optim.Adam(self.model.parameters(), lr=ec["learning_rate"])
            rng = np.random.default_rng(seed)
            for _ in range(ec["neural_epochs"]):
                for ids in np.array_split(
                    rng.permutation(len(x)),
                    max(1, int(np.ceil(len(x) / ec["batch_size"]))),
                ):
                    opt.zero_grad()
                    p = self.model(a[ids])
                    loss = ((p - b[ids])[v[ids]] ** 2).mean()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
                    opt.step()
            self.model.eval()
        elif kind == "chronos":
            from chronos import ChronosPipeline

            rev = ec.get("chronos_revision")
            if not rev or len(rev) != 40:
                raise ValueError(
                    "Chronos requires a frozen 40-character checkpoint revision"
                )
            self.model = ChronosPipeline.from_pretrained(
                ec["chronos_model"], revision=rev, torch_dtype=torch.float32
            )
            self.model.model.to(pool.device)
        return self

    def predict(self, pool, i, origin, history):
        if self.kind not in EXTRA:
            self.model.panel = pool.panel
            self.model.device = pool.device
            self.model.fill = pool.fill
            try:
                return self.model.predict(self.kind, i, origin)
            finally:
                self.model.panel = None
        if self.kind == "catboost":
            from fame_scm.experts.forecast_experts import tree_rows

            return self.model.predict(
                np.c_[
                    tree_rows(
                        history,
                        pool.panel.dates[origin],
                        pool.h,
                        pool.cfg["data"].get("frequency", "D"),
                    ),
                    pool.context_rows(i, origin),
                ]
            )
        if self.kind in NF_NAMES:
            df = pd.DataFrame(
                {
                    "unique_id": ["0"] * len(history),
                    "ds": pd.date_range(
                        end=pool.panel.dates[origin],
                        periods=len(history),
                        freq=pool.cfg["data"].get("frequency", "D"),
                    ),
                    "y": history,
                }
            )
            # Fixed request-derived seed: independent of call order and cache hits.
            seed = self.request_seed(pool, i, origin)
            with torch.random.fork_rng(
                devices=[] if pool.device.type == "cpu" else [pool.device.index or 0]
            ):
                torch.manual_seed(seed)
                result = self.model.predict(df=df)
            cols = [c for c in result if c not in ["unique_id", "ds"]]
            mean = [c for c in cols if c.endswith("-mean")]
            col = mean[0] if mean else NF_NAMES[self.kind]
            if col not in result:
                raise ValueError(f"Missing expected point forecast {col}: {cols}")
            return result[col].to_numpy()
        if self.kind == "mamba":
            scale = max(np.abs(history).mean(), 1.0)
            with torch.no_grad():
                return (
                    self.model(
                        torch.tensor(
                            history / scale, dtype=torch.float32, device=pool.device
                        )[None]
                    )[0]
                    .cpu()
                    .numpy()
                    * scale
                )
        with torch.random.fork_rng(
            devices=[] if pool.device.type == "cpu" else [pool.device.index or 0]
        ):
            torch.manual_seed(self.request_seed(pool, i, origin))
            pred = self.model.predict(
                torch.tensor(history, dtype=torch.float32)[None],
                prediction_length=pool.h,
                num_samples=self.params.get("num_samples", 20),
            )
        return pred[0].mean(0).cpu().numpy()

    def request_seed(self, pool, i, origin):
        key = f'{pool.cfg["seed"]}:{self.name}:{pool.panel.ids[i]}:{origin}'
        return int(hashlib.sha256(key.encode()).hexdigest()[:8], 16) % (2**31)

    def cpu(self):
        return self.to("cpu")

    def to(self, device):
        from fame_scm.devices import prepare_device

        device = prepare_device(device)
        if isinstance(self.model, nn.Module):
            self.model.to(device).eval()
        elif self.kind == "chronos":
            self.model.model.to(device).eval()
        elif self.kind in NF_NAMES:
            for m in self.model.models:
                m.to(device).eval()
                if hasattr(m, "trainer_kwargs"):
                    m.trainer_kwargs.update(
                        accelerator="gpu" if device.type == "cuda" else "cpu",
                        devices=[device.index] if device.type == "cuda" else 1,
                    )
        elif self.kind not in EXTRA:
            self.model.to(device)
        return self
