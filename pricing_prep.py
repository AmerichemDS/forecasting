"""Pricing data preparation pipeline for monthly Customer x Product analysis."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd


COLS: Dict[str, Optional[str]] = {
    "global_id": "Global ID",
    "cust_nbr": "ROSS_CUST_NBR",
    "dollars": "DOLLARS (Invoiced Amt) USD",
    "pounds": "POUNDS (Invoice Qty) LB",
    "gl_date": "GL_Date",
    "sbu": None,
    "plant": None,
}


DEFAULT_ALIASES: Dict[str, Tuple[str, ...]] = {
    "global_id": ("Global ID", "GlobalID", "Product ID"),
    "cust_nbr": ("ROSS_CUST_NBR", "Cust No.", "Customer Number", "Customer"),
    "dollars": ("DOLLARS (Invoiced Amt) USD", "Invoice Dollars", "Amount USD", "Amount"),
    "pounds": ("POUNDS (Invoice Qty) LB", "Invoice Pounds", "Qty LB", "Pounds"),
    "gl_date": ("GL_Date", "Invoice Date", "GL Date", "Date"),
    "sbu": ("SBU",),
    "plant": ("Plant",),
}


def _standardize_name(name: Any) -> str:
    """Normalize raw column names to make mapping robust."""
    return " ".join(str(name).strip().split())


def _resolve_columns(df: pd.DataFrame, cols: Dict[str, Optional[str]]) -> Dict[str, str]:
    """Resolve canonical column names against dataframe columns using explicit map + aliases."""
    normalized = {_standardize_name(c): c for c in df.columns}
    resolved: Dict[str, str] = {}

    for canonical in ("global_id", "cust_nbr", "dollars", "pounds", "gl_date", "sbu", "plant"):
        requested = cols.get(canonical)
        if requested:
            candidate = _standardize_name(requested)
            if candidate in normalized:
                resolved[canonical] = normalized[candidate]
                continue

        for alias in DEFAULT_ALIASES.get(canonical, ()):  # fallback
            candidate = _standardize_name(alias)
            if candidate in normalized:
                resolved[canonical] = normalized[candidate]
                break

    required = {"global_id", "cust_nbr", "dollars", "pounds", "gl_date"}
    missing = sorted(required - set(resolved))
    if missing:
        raise ValueError(f"Missing required columns after mapping: {missing}")

    return resolved


def load_sales_excel(
    excel_path: str | Path,
    sheet_name: Optional[str] = None,
    header: int = 0,
    dtype: Optional[Dict[str, str]] = None,
    cols: Optional[Dict[str, Optional[str]]] = None,
) -> Tuple[pd.DataFrame, Dict[str, str]]:
    """Load, map, and clean sales data from Excel into DataFrame `df`."""
    col_map = cols or COLS
    df = pd.read_excel(excel_path, sheet_name=sheet_name, header=header, dtype=dtype)
    df.columns = [_standardize_name(c) for c in df.columns]

    resolved = _resolve_columns(df, col_map)
    rename_to_business = {
        resolved["global_id"]: "Global ID",
        resolved["cust_nbr"]: "ROSS_CUST_NBR",
        resolved["dollars"]: "DOLLARS (Invoiced Amt) USD",
        resolved["pounds"]: "POUNDS (Invoice Qty) LB",
        resolved["gl_date"]: "GL_Date",
    }
    if "sbu" in resolved:
        rename_to_business[resolved["sbu"]] = "SBU"
    if "plant" in resolved:
        rename_to_business[resolved["plant"]] = "Plant"

    df = df.rename(columns=rename_to_business)

    for numeric_col in ["DOLLARS (Invoiced Amt) USD", "POUNDS (Invoice Qty) LB"]:
        df[numeric_col] = pd.to_numeric(df[numeric_col], errors="coerce")

    invalid_date_mask = pd.to_datetime(df["GL_Date"], errors="coerce").isna()
    invalid_date_count = int(invalid_date_mask.sum())
    if invalid_date_count > 0:
        print(f"[WARN] Invalid GL_Date rows dropped: {invalid_date_count}")

    df["GL_Date"] = pd.to_datetime(df["GL_Date"], errors="coerce")
    df = df.dropna(subset=["GL_Date"]).copy()

    df["DOLLARS (Invoiced Amt) USD"] = df["DOLLARS (Invoiced Amt) USD"].fillna(0.0)
    df["POUNDS (Invoice Qty) LB"] = df["POUNDS (Invoice Qty) LB"].fillna(0.0)

    return df, resolved


def _mode_with_warning(series: pd.Series, label: str) -> Any:
    """Return mode; warn when non-unique values appear."""
    non_null = series.dropna()
    if non_null.empty:
        return np.nan
    nunique = non_null.nunique()
    if nunique > 1:
        print(f"[WARN] Non-unique {label} in series; choosing mode.")
    modes = non_null.mode(dropna=True)
    return modes.iloc[0] if not modes.empty else non_null.iloc[0]


def build_pricing_monthly(df: pd.DataFrame) -> pd.DataFrame:
    """Build complete monthly customer-product table and fill prices for delta analysis."""
    work = df.copy()
    work["Month"] = work["GL_Date"].dt.to_period("M").dt.to_timestamp()

    agg = (
        work.groupby(["Global ID", "ROSS_CUST_NBR", "Month"], as_index=False)
        .agg(
            SumDollars_USD=("DOLLARS (Invoiced Amt) USD", "sum"),
            SumPounds_LB=("POUNDS (Invoice Qty) LB", "sum"),
        )
        .sort_values(["Global ID", "ROSS_CUST_NBR", "Month"])
    )
    agg["Price_USD_per_LB"] = np.where(
        agg["SumPounds_LB"].eq(0),
        np.nan,
        agg["SumDollars_USD"] / agg["SumPounds_LB"],
    )

    pairs = agg[["Global ID", "ROSS_CUST_NBR"]].drop_duplicates()
    full_frames = []
    for gid, cust in pairs.itertuples(index=False):
        subset = agg[(agg["Global ID"] == gid) & (agg["ROSS_CUST_NBR"] == cust)].set_index("Month")
        idx = pd.date_range(subset.index.min(), subset.index.max(), freq="MS")
        reindexed = subset.reindex(idx)
        reindexed["Global ID"] = gid
        reindexed["ROSS_CUST_NBR"] = cust
        full_frames.append(reindexed.reset_index().rename(columns={"index": "Month"}))

    pricing_monthly = pd.concat(full_frames, ignore_index=True)
    pricing_monthly = pricing_monthly.sort_values(["Global ID", "ROSS_CUST_NBR", "Month"]).reset_index(drop=True)

    no_sales_mask = pricing_monthly["SumPounds_LB"].isna()
    pricing_monthly.loc[no_sales_mask, "SumDollars_USD"] = 0.0
    pricing_monthly.loc[no_sales_mask, "SumPounds_LB"] = 0.0

    pricing_monthly["Price_USD_per_LB"] = np.where(
        pricing_monthly["SumPounds_LB"].eq(0),
        np.nan,
        pricing_monthly["SumDollars_USD"] / pricing_monthly["SumPounds_LB"],
    )

    pricing_monthly["Price_Filled_USD_per_LB"] = (
        pricing_monthly.groupby(["Global ID", "ROSS_CUST_NBR"], group_keys=False)["Price_USD_per_LB"]
        .apply(lambda s: s.ffill().bfill())
    )

    all_nan_series = (
        pricing_monthly.groupby(["Global ID", "ROSS_CUST_NBR"])["Price_Filled_USD_per_LB"]
        .apply(lambda s: s.isna().all())
        .sum()
    )
    pricing_monthly["All_NaN_Price_Series_Flag"] = pricing_monthly.groupby(
        ["Global ID", "ROSS_CUST_NBR"]
    )["Price_Filled_USD_per_LB"].transform(lambda s: s.isna().all())

    dupes = pricing_monthly.duplicated(subset=["Global ID", "ROSS_CUST_NBR", "Month"]).sum()
    assert dupes == 0, "Duplicate keys found at (Global ID, ROSS_CUST_NBR, Month) grain"

    print(f"[INFO] Series with all-NaN filled price: {int(all_nan_series)}")

    return pricing_monthly[
        [
            "Global ID",
            "ROSS_CUST_NBR",
            "Month",
            "SumDollars_USD",
            "SumPounds_LB",
            "Price_USD_per_LB",
            "Price_Filled_USD_per_LB",
            "All_NaN_Price_Series_Flag",
        ]
    ]


def attach_optional_dimensions(raw_df: pd.DataFrame, pricing_monthly: pd.DataFrame) -> pd.DataFrame:
    """Attach optional SBU/Plant using mode at customer-product level."""
    optional_cols = [c for c in ["SBU", "Plant"] if c in raw_df.columns]
    if not optional_cols:
        return pricing_monthly

    attrs = (
        raw_df.groupby(["Global ID", "ROSS_CUST_NBR"], as_index=False)[optional_cols]
        .agg({c: (lambda x, col=c: _mode_with_warning(x, col)) for c in optional_cols})
    )

    return pricing_monthly.merge(attrs, on=["Global ID", "ROSS_CUST_NBR"], how="left")


def add_pricing_views(pricing_monthly: pd.DataFrame) -> pd.DataFrame:
    """Add MoM, YoY, and YTD-vs-PYTD pricing change columns."""
    pricing_views = pricing_monthly.sort_values(["Global ID", "ROSS_CUST_NBR", "Month"]).copy()
    grp = pricing_views.groupby(["Global ID", "ROSS_CUST_NBR"])

    pricing_views["Prev_Month_Price"] = grp["Price_Filled_USD_per_LB"].shift(1)
    pricing_views["MoM_Abs_Change"] = pricing_views["Price_Filled_USD_per_LB"] - pricing_views["Prev_Month_Price"]
    pricing_views["MoM_Pct_Change"] = np.where(
        pricing_views["Prev_Month_Price"].isna() | pricing_views["Prev_Month_Price"].eq(0),
        np.nan,
        pricing_views["MoM_Abs_Change"] / pricing_views["Prev_Month_Price"],
    )

    pricing_views["Prev_Year_Month_Price"] = grp["Price_Filled_USD_per_LB"].shift(12)
    pricing_views["YoY_Abs_Change"] = pricing_views["Price_Filled_USD_per_LB"] - pricing_views["Prev_Year_Month_Price"]
    pricing_views["YoY_Pct_Change"] = np.where(
        pricing_views["Prev_Year_Month_Price"].isna() | pricing_views["Prev_Year_Month_Price"].eq(0),
        np.nan,
        pricing_views["YoY_Abs_Change"] / pricing_views["Prev_Year_Month_Price"],
    )

    pricing_views["Year"] = pricing_views["Month"].dt.year
    pricing_views["MonthNum"] = pricing_views["Month"].dt.month
    pricing_views["YTD_Dollars"] = grp["SumDollars_USD"].cumsum()
    pricing_views["YTD_Pounds"] = grp["SumPounds_LB"].cumsum()
    pricing_views["YTD_Avg_Price"] = np.where(
        pricing_views["YTD_Pounds"].eq(0),
        np.nan,
        pricing_views["YTD_Dollars"] / pricing_views["YTD_Pounds"],
    )

    pricing_views["PYTD_Avg_Price"] = grp["YTD_Avg_Price"].shift(12)
    pricing_views["YTD_Abs_Change"] = pricing_views["YTD_Avg_Price"] - pricing_views["PYTD_Avg_Price"]
    pricing_views["YTD_Pct_Change"] = np.where(
        pricing_views["PYTD_Avg_Price"].isna() | pricing_views["PYTD_Avg_Price"].eq(0),
        np.nan,
        pricing_views["YTD_Abs_Change"] / pricing_views["PYTD_Avg_Price"],
    )

    return pricing_views


def print_quality_summary(pricing_monthly: pd.DataFrame) -> None:
    """Print quality checks and high-level coverage metrics."""
    pair_cols = ["Global ID", "ROSS_CUST_NBR"]
    total_rows = len(pricing_monthly)
    no_sales_pct = pricing_monthly["SumPounds_LB"].eq(0).mean() * 100
    missing_price_pct = pricing_monthly["Price_USD_per_LB"].isna().mean() * 100
    missing_filled_pct = pricing_monthly["Price_Filled_USD_per_LB"].isna().mean() * 100
    all_nan_series_count = (
        pricing_monthly.groupby(pair_cols)["Price_Filled_USD_per_LB"].apply(lambda s: s.isna().all()).sum()
    )

    print("\n===== Pricing Monthly Quality Summary =====")
    print(f"Unique customers: {pricing_monthly['ROSS_CUST_NBR'].nunique():,}")
    print(f"Unique products: {pricing_monthly['Global ID'].nunique():,}")
    print(f"Unique customer-product pairs: {pricing_monthly[pair_cols].drop_duplicates().shape[0]:,}")
    print(f"Total rows (months): {total_rows:,}")
    print(f"Distinct Month count: {pricing_monthly['Month'].nunique():,}")
    print(f"% months with no sales (SumPounds_LB == 0): {no_sales_pct:.2f}%")
    print(f"% missing Price_USD_per_LB before fill: {missing_price_pct:.2f}%")
    print(f"% missing Price_Filled_USD_per_LB after fill: {missing_filled_pct:.2f}%")
    print(f"Series with all-NaN filled price: {int(all_nan_series_count):,}")


def run_pricing_pipeline(
    excel_path: str | Path,
    sheet_name: Optional[str] = None,
    header: int = 0,
    dtype: Optional[Dict[str, str]] = None,
    cols: Optional[Dict[str, Optional[str]]] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run full ingest + monthly prep + pricing views pipeline.

    Returns:
        df: cleaned raw dataframe
        pricing_monthly: base monthly table
        pricing_views: monthly table with pricing change columns
    """
    df, _ = load_sales_excel(
        excel_path=excel_path,
        sheet_name=sheet_name,
        header=header,
        dtype=dtype,
        cols=cols,
    )
    pricing_monthly = build_pricing_monthly(df)
    pricing_monthly = attach_optional_dimensions(df, pricing_monthly)
    pricing_views = add_pricing_views(pricing_monthly)
    print_quality_summary(pricing_monthly)
    return df, pricing_monthly, pricing_views


if __name__ == "__main__":
    # Example usage
    excel_path = "./sales_input.xlsx"
    sheet_name = None  # Example: "Sheet1"

    df, pricing_monthly, pricing_views = run_pricing_pipeline(
        excel_path=excel_path,
        sheet_name=sheet_name,
        header=0,
        dtype=None,
        cols=COLS,
    )

    print("\nBase table: pricing_monthly")
    print(pricing_monthly.head())
    print("\nDerived table: pricing_views")
    print(pricing_views.head())
