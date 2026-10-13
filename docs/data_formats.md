# Data formats

## Daily-sales CSV

| Field | Type | Role |
| --- | --- | --- |
| `date` | Date | Daily time index |
| `vem_id` | String | Terminal identity |
| `merc_id` | String | Product identity |
| `daily_quantity` | Nonnegative number | Forecast target |
| `merc_sale_price`, `coupon_amount`, `capacity` | Number | Optional origin-known numeric context |
| `merc_brand_code`, `merc_type_code`, `machine_type`, `scene_code`, `city_name` | String | Optional categorical context |
| Configured availability column | Boolean-like | Known executable/uncensored sales mask |

Column names are configured under `data`. Duplicate entity/date records use the
configured `first`, `sum`, or `error` policy. `first` retains the first source row
before sorting. Negative targets use `error` or `clip_zero`.

`missing_dates: zero` marks absent days after the first record as valid zero
sales. `unknown` leaves them masked. A configured availability column overrides
that fill policy; missing availability is invalid. Metadata is forward-filled.
`capacity=0` alone does not imply a stockout. `daily_sales_amount` is not a target
or a fingerprint input.

The schema example at [`data/daily_sales_format_example.csv`](../data/daily_sales_format_example.csv)
is a field-format example; its short history is not an experiment cohort.

## Hourly wide CSV

The first declared timestamp column is followed by numeric meter/sensor columns:

```csv
date,meter_0,meter_1
2024-01-01 00:00:00,12.4,7.2
2024-01-01 01:00:00,13.1,7.5
```

Timestamps must be unique, timezone-naive, and aligned to `data.frequency`.
Missing values remain masked. Missing grid timestamps require
`missing_timestamps: reindex`; the default rejects them. `value_columns` selects
an explicit column list, while `drop_columns` excludes fields from automatic
selection.

`standardization: train_zscore` uses frozen training means/scales and requires
`forecast.clip_nonnegative: false`. Inference requires the saved scaler for
each fitted series. Fingerprints restore original signal units before extracting
temporal descriptors.

## Prepared retail panel

`prepare-retail` writes NumPy arrays and JSON metadata. `load_retail_panel`
verifies the preparation cutoff and, when configured, array checksums. Series
are store–item pairs. Sales, validity, calendar context, static attributes, and
numeric context retain their original time grid.

M5 first listing uses historical price availability. Favorita training-qualified
IDs are selected before test-period sales are considered. Favorita absent
records after first appearance use the declared zero-sales convention. Price
data and observed-sales proxies do not independently identify all stockouts.
Explicit inventory masks provide censoring information.

## Inventory replay inputs

Forecast CSV fields:

| Field | Role |
| --- | --- |
| `entity_id` | Series identity matching the daily file |
| `origin` | Forecast issue date |
| `target_date` | Predicted date |
| `prediction` | Nonnegative forecast |

Daily state CSV fields:

| Field | Role |
| --- | --- |
| `entity_id`, `date` | Unique, contiguous daily series index |
| `sales` | Nonnegative observed-sales proxy |
| `available` | Executable trading-day indicator |
| `uncensored` | Indicator allowing sales to contribute to replay demand |
| `capacity` | Nonnegative slot capacity |
| `initial_stock` | Initial state for the entity |

The inventory runner uses `lead_days`, `review_days`, `forecast_horizon`, and
asymmetric loss weights from the policy YAML. Forecast origin must precede the
order decision; later forecasts never fill an earlier missing forecast.
