import unittest, tempfile
from pathlib import Path
import numpy as np
import pandas as pd
from fame_scm.evaluation.inventory import replay_inventory
from fame_scm.experts.pools import configure_pool
from fame_scm.settings import load_settings


class DeploymentContracts(unittest.TestCase):
    def fixtures(self):
        dates = pd.date_range("2024-01-01", periods=4)
        daily = pd.DataFrame(
            {
                "entity_id": ["a"] * 4,
                "date": dates,
                "sales": [2, 3, 8, 1],
                "available": [True] * 4,
                "uncensored": [True] * 4,
                "capacity": [10] * 4,
                "initial_stock": [5] * 4,
            }
        )
        forecasts = []
        for o in pd.date_range("2023-12-31", periods=5):
            for h in range(1, 5):
                forecasts.append(
                    {
                        "entity_id": "a",
                        "origin": o,
                        "target_date": o + pd.Timedelta(days=h),
                        "prediction": 2.0,
                    }
                )
        policy = {"lead_days": 1, "review_days": 1, "forecast_horizon": 4}
        return pd.DataFrame(forecasts), daily, policy

    def test_pool_cardinality_unique_variants(self):
        for n in [5, 10, 15, 20, 40]:
            c = configure_pool(load_settings(), n)
            self.assertEqual(len(set(c["experts"]["names"])), n)
        c = configure_pool(load_settings(), 40)
        self.assertTrue(
            all(
                c["experts"]["specs"][n]["parameters"]
                for n in c["experts"]["names"]
                if "__variant" in n
            )
        )

    def test_inventory_conservation_and_lead_time(self):
        p, d, c = self.fixtures()
        r, s = replay_inventory(p, d, c)
        np.testing.assert_allclose(
            r.opening_stock + r.arrivals - r.capacity_discard - r.fulfilled_proxy,
            r.closing_stock,
        )
        self.assertEqual(r.iloc[0].arrivals, 0)
        self.assertEqual(r.iloc[1].arrivals, r.iloc[0].order)
        self.assertTrue((r.closing_stock >= 0).all())
        self.assertTrue((r.closing_stock + r.in_transit <= 10 + 1e-9).all())

    def test_future_forecasts_cannot_affect_earlier_orders(self):
        p, d, c = self.fixtures()
        r, _ = replay_inventory(p, d, c)
        p.loc[p.origin >= pd.Timestamp("2024-01-03"), "prediction"] = 100
        changed, _ = replay_inventory(p, d, c)
        np.testing.assert_array_equal(r.order[:2], changed.order[:2])

    def test_missing_forecast_fails_not_future_filled(self):
        p, d, c = self.fixtures()
        p = p[p.origin > pd.Timestamp("2023-12-31")]
        with self.assertRaisesRegex(ValueError, "causal"):
            replay_inventory(p, d, c)

    def test_unavailable_days_not_demand_or_depletion(self):
        p, d, c = self.fixtures()
        d.loc[2, "available"] = False
        r, s = replay_inventory(p, d, c)
        self.assertEqual(r.iloc[2].sales_proxy, 0)
        self.assertFalse(r.iloc[2].depletion_event)

    def test_forecast_duplicate_and_negative_rejected(self):
        p, d, c = self.fixtures()
        with self.assertRaises(ValueError):
            replay_inventory(pd.concat([p, p.iloc[:1]]), d, c)
        p.loc[0, "prediction"] = -1
        with self.assertRaises(ValueError):
            replay_inventory(p, d, c)

    def test_missing_calendar_day_rejected(self):
        p, d, c = self.fixtures()
        with self.assertRaisesRegex(ValueError, "contiguous"):
            replay_inventory(p, d.drop(index=1), c)

    def test_selective_ssm_input_dependent_and_finite_gradient(self):
        import torch
        from fame_scm.experts.extended_experts import SelectiveSSM

        torch.manual_seed(5)
        m = SelectiveSSM(3, width=4, state_size=2)
        x = torch.randn(2, 8, requires_grad=True)
        y = m(x)
        self.assertEqual(tuple(y.shape), (2, 3))
        y.square().mean().backward()
        self.assertTrue(torch.isfinite(x.grad).all())
        self.assertGreater(x.grad.abs().sum(), 0)
