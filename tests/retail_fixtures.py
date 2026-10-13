"""Synthetic schema fixtures, explicitly NOT public benchmark observations."""

from pathlib import Path
import numpy as np
import pandas as pd


def make_retail_fixtures(root, n=16, days=700):
    root = Path(root)
    m5 = root / "m5"
    fav = root / "favorita"
    m5.mkdir(parents=True)
    fav.mkdir(parents=True)
    dates = pd.date_range("2015-01-01", periods=days)
    rng = np.random.default_rng(17)
    weeks = np.arange(days) // 7
    calendar = pd.DataFrame(
        {
            "d": [f"d_{x+1}" for x in range(days)],
            "date": dates,
            "wm_yr_wk": weeks,
            "event_name_1": ["holiday" if x % 31 == 0 else None for x in range(days)],
            "event_name_2": None,
            "snap_CA": (np.arange(days) % 30 < 10).astype(int),
        }
    )
    calendar.to_csv(m5 / "calendar.csv", index=False)
    sales = []
    prices = []
    transactions = []
    items = []
    for i in range(n):
        y = rng.poisson(2 + i % 4, size=days)
        row = {
            "id": f"item_{i}_CA_1_evaluation",
            "item_id": f"item_{i}",
            "store_id": "CA_1",
            "state_id": "CA",
            "dept_id": f"dept_{i%2}",
            "cat_id": "FOODS",
        }
        row.update({f"d_{j+1}": int(v) for j, v in enumerate(y)})
        sales.append(row)
        prices.extend(
            {
                "store_id": "CA_1",
                "item_id": f"item_{i}",
                "wm_yr_wk": int(w),
                "sell_price": float(2 + i % 3),
            }
            for w in np.unique(weeks)
        )
        items.append(
            {
                "item_nbr": i,
                "family": f"family_{i%3}",
                "class": i % 2,
                "perishable": i % 2,
            }
        )
        for j, d in enumerate(dates):
            # Missing transaction means zero under the explicit fixture policy.
            if j % 11 == 0 and j > 0:
                continue
            transactions.append(
                {
                    "id": len(transactions),
                    "date": d,
                    "store_nbr": 1,
                    "item_nbr": i,
                    "unit_sales": -2 if (i == 0 and j == 2) else int(y[j]),
                    "onpromotion": bool(j % 10 == 0),
                }
            )
    pd.DataFrame(sales).to_csv(m5 / "sales_train_evaluation.csv", index=False)
    pd.DataFrame(prices).to_csv(m5 / "sell_prices.csv", index=False)
    pd.DataFrame(transactions).to_csv(fav / "train.csv", index=False)
    pd.DataFrame(items).to_csv(fav / "items.csv", index=False)
    pd.DataFrame(
        [
            {
                "store_nbr": 1,
                "city": "Quito",
                "state": "Pichincha",
                "type": "A",
                "cluster": 1,
            }
        ]
    ).to_csv(fav / "stores.csv", index=False)
    pd.DataFrame(
        {"date": dates[::2], "dcoilwtico": np.arange(len(dates[::2])) * 0.01 + 40}
    ).to_csv(fav / "oil.csv", index=False)
    pd.DataFrame(
        [
            {
                "date": dates[10],
                "type": "Holiday",
                "locale": "National",
                "locale_name": "Ecuador",
                "description": "fixture",
                "transferred": False,
            },
            {
                "date": dates[12],
                "type": "Holiday",
                "locale": "Local",
                "locale_name": "Guayaquil",
                "description": "wrong locale",
                "transferred": False,
            },
            {
                "date": dates[13],
                "type": "Holiday",
                "locale": "National",
                "locale_name": "Ecuador",
                "description": "transferred",
                "transferred": True,
            },
        ]
    ).to_csv(fav / "holidays_events.csv", index=False)
    return m5, fav, dates
