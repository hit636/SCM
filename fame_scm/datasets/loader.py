"""Select the panel reader for a declared dataset format."""


def load_panel(cfg):
    kind = cfg["data"].get("format", "snbc")
    if kind == "wide_csv":
        from .wide import load_wide_panel

        return load_wide_panel(cfg)
    if kind == "retail_panel":
        from .retail import load_retail_panel

        return load_retail_panel(cfg)
    if kind == "snbc":
        from .snbc import load_panel as load_sales_panel

        return load_sales_panel(cfg)
    raise ValueError(f"Unknown data format: {kind}")
