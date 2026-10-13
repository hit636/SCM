import tempfile, unittest
from pathlib import Path
import numpy as np
import torch
from fame_scm.settings import load_settings
from fame_scm.datasets.snbc import make_demo, load_panel, read_sales
from fame_scm.fingerprints.features import Forecastability
from fame_scm.competence.torch_models import CompetenceModel, SparseRouter
from fame_scm.competence.torch_objectives import (
    competence_objective,
    sparse_router_objective,
)


class TrainingContracts(unittest.TestCase):
    def test_zero_sales_fill_counts_calendar_training_days(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sales.csv"
            path.write_text(
                "date,vem_id,merc_id,daily_quantity\n"
                "2024/1/1,a,b,2\n"
                "2024/7/1,a,b,\n"
                "2025/2/23,a,b,1\n"
                "2024/1/1,a,b,99\n"
                "2025/2/23,a,late,1\n"
            )
            cfg = load_settings()
            cfg["data"]["path"] = str(path)
            panel = load_panel(cfg)
            self.assertEqual(len(panel.ids), 1)
            self.assertEqual(panel.y[0, 0], 2)
            self.assertEqual(panel.y[0, 1], 0)
            self.assertTrue(panel.valid[0].all())
            self.assertEqual(panel.audit["blank_target_rows_filled_zero"], 1)
            self.assertEqual(panel.audit["dropped_duplicate_rows"], 1)
            self.assertEqual(panel.audit["training_observed_days_scanned"]["max"], 2)
            self.assertEqual(panel.audit["training_valid_days_scanned"]["max"], 294)
            self.assertEqual(
                panel.audit["exclusion_reason_counts"][
                    "insufficient_valid_training_days"
                ],
                1,
            )
            cfg["data"]["missing_dates"] = "unknown"
            with self.assertRaisesRegex(ValueError, "target contains nulls"):
                read_sales(cfg)

    def test_sparse_sales_exclusions_report_training_and_stage_coverage(self):
        import pandas as pd

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sales.csv"
            make_demo(path, series=2, days=420)
            sparse = pd.read_csv(path)
            sparse = sparse[
                pd.to_datetime(sparse["date"], format="mixed").dt.dayofweek == 0
            ]
            sparse.to_csv(path, index=False)
            cfg = load_settings()
            cfg["data"]["path"] = str(path)
            cfg["output"] = str(Path(directory) / "run")
            cfg["data"]["missing_dates"] = "unknown"
            cfg["data"]["min_observed_train_days"] = 49
            with self.assertRaisesRegex(ValueError, "No eligible series among 2"):
                load_panel(cfg)
            import json

            audit = json.loads((Path(cfg["output"]) / "data_audit.json").read_text())
            self.assertEqual(
                audit["exclusion_reason_counts"]["insufficient_training_observations"],
                2,
            )
            self.assertEqual(
                audit["exclusion_reason_counts"]["test_target_coverage"], 2
            )
            self.assertEqual(
                audit["eligibility_requirements"]["min_valid_targets_per_origin"], 12
            )
            self.assertEqual(
                audit["excluded_examples"][0]["training_observed_days"], 42
            )
            cfg["data"]["missing_dates"] = "zero"
            with self.assertRaisesRegex(ValueError, "No eligible series among 2"):
                load_panel(cfg)
            audit = json.loads((Path(cfg["output"]) / "data_audit.json").read_text())
            self.assertEqual(
                audit["exclusion_reason_counts"]["insufficient_training_observations"],
                2,
            )
            self.assertEqual(
                audit["exclusion_reason_counts"]["test_target_coverage"], 0
            )
            cfg["data"]["min_observed_train_days"] = 20
            self.assertEqual(len(load_panel(cfg).ids), 2)

    def test_duplicate_daily_rows_keep_first_source_record(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sales.csv"
            path.write_text(
                "date,vem_id,merc_id,daily_quantity,scene_code\n"
                "2025/10/2,a,b,2,first\n"
                "2025/10/1,a,b,1,earlier_day\n"
                "2025-10-02 18:00:00,a,b,99,later_duplicate\n"
                "2025/10/2,a,c,3,different_product\n"
            )
            cfg = load_settings()
            cfg["data"]["path"] = str(path)
            frame, audit = read_sales(cfg)
            self.assertEqual(audit["input_rows"], 4)
            self.assertEqual(audit["output_rows"], 3)
            self.assertEqual(audit["dropped_duplicate_rows"], 1)
            retained = frame[
                (frame["merc_id"] == "b") & (frame["_date"].dt.day == 2)
            ].iloc[0]
            self.assertEqual(retained["_y"], 2)
            self.assertEqual(retained["scene_code"], "first")
            cfg["data"]["duplicates"] = "error"
            with self.assertRaisesRegex(ValueError, "duplicate entity/date"):
                read_sales(cfg)
            cfg["data"]["duplicates"] = "sum"
            summed, _ = read_sales(cfg)
            self.assertEqual(
                summed[(summed["merc_id"] == "b") & (summed["_date"].dt.day == 2)].iloc[
                    0
                ]["_y"],
                101,
            )

    def test_hidden_labels_cannot_affect_scm_loss_or_gradient(self):
        torch.manual_seed(2)
        model = CompetenceModel(3, 2, 3)
        x = torch.randn(2, 3)
        ids = torch.arange(2)
        mask = torch.tensor([[1, 0, 1], [1, 1, 0]], dtype=torch.bool)
        labels = torch.tensor([[1.0, float("nan"), 2.0], [3.0, 2.0, float("nan")]])
        output = model(x, ids)
        first, _ = competence_objective(
            model,
            output,
            labels,
            mask,
            rank_margin=0.05,
            lambda_factor=0,
            lambda_encoder=0,
        )
        alternate = labels.clone()
        alternate[~mask] = 1e6
        second, _ = competence_objective(
            model,
            output,
            alternate,
            mask,
            rank_margin=0.05,
            lambda_factor=0,
            lambda_encoder=0,
        )
        self.assertAlmostEqual(float(first.detach()), float(second.detach()))
        first.backward()
        self.assertTrue(
            all(
                torch.isfinite(p.grad).all()
                for p in model.parameters()
                if p.grad is not None
            )
        )

    def test_single_observed_forecast_has_zero_prediction_gradient(self):
        logits = torch.tensor([[0.2, -0.1, 0.8]], requires_grad=True)
        obs = torch.tensor([[True, False, False]])
        cache = torch.tensor(
            [[[1.0, 2.0], [float("nan"), float("nan")], [float("nan"), float("nan")]]]
        )
        loss, _ = sparse_router_objective(
            logits,
            torch.tensor([[0.3, 0.3, 0.4]]),
            cache,
            obs,
            torch.tensor([[2.0, 3.0]]),
            torch.tensor([[True, True]]),
            torch.ones(3),
            prediction_loss="mse",
            lambda_teacher=0.0,
            lambda_balance=0.0,
            lambda_cost=0.0,
        )
        loss.backward()
        np.testing.assert_allclose(logits.grad.numpy(), 0, atol=1e-7)

    def test_appended_head_freezes_old_parameters_and_dropout(self):
        model = SparseRouter(4, 3)
        old = list(model.parameters())
        new = model.freeze_old_and_append()
        model.train_appended_head_only()
        self.assertTrue(all(not p.requires_grad for p in old))
        self.assertTrue(all(p.requires_grad for p in new))
        self.assertFalse(model.encoder.training)

    def test_real_format_and_future_metadata_isolation(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sales.csv"
            make_demo(path, series=4, days=420)
            cfg = load_settings()
            cfg["data"]["path"] = str(path)
            panel = load_panel(cfg)
            extractor = Forecastability(cfg).fit(panel)
            end = panel.stages["train"].end
            before = extractor.transform(panel, end)
            for frame in panel.frames:
                frame.loc[panel.dates[end + 1] :, "merc_sale_price"] = "99999"
            after = extractor.transform(panel, end)
            np.testing.assert_array_equal(before, after)
            self.assertEqual(len(panel.ids), 4)

    def test_three_rows_cannot_be_silently_expanded_into_training_data(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sales.csv"
            path.write_text(
                "date,vem_id,merc_id,daily_quantity\n2025/9/24,a,b,1\n2025/10/14,a,b,2\n2025/11/5,a,b,1\n"
            )
            cfg = load_settings()
            cfg["data"]["path"] = str(path)
            with self.assertRaisesRegex(ValueError, "Insufficient data"):
                load_panel(cfg)


if __name__ == "__main__":
    unittest.main()
