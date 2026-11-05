"""Data ingestion and reproducible demonstration dataset (NOT real business observations)."""
from pathlib import Path
import numpy as np
import pandas as pd

REQUIRED = {"country", "date", "revenue"}


def make_demo_data(path, seed=42):
    """Create clearly-labeled synthetic monthly revenue data for a runnable example."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2020-01-01", periods=60, freq="MS")
    base = {"United States": 10000, "Canada": 6500, "United Kingdom": 8000}
    records = []
    for country, offset in base.items():
        for i, date in enumerate(dates):
            seasonal = 700 * np.sin(2 * np.pi * (date.month - 1) / 12)
            revenue = max(0, offset + i * 75 + seasonal + rng.normal(0, 150))
            records.append({"country": country, "date": date.strftime("%Y-%m-%d"), "revenue": round(revenue, 2)})
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_csv(path, index=False)
    return path


def ingest(path):
    """Validate input and return standardized monthly country and ALL aggregates."""
    df = pd.read_csv(path)
    if not REQUIRED.issubset(df.columns):
        raise ValueError("CSV must contain country, date, revenue")
    df = df[["country", "date", "revenue"]].copy()
    df["date"] = pd.to_datetime(df["date"], errors="raise")
    df["revenue"] = pd.to_numeric(df["revenue"], errors="raise")
    if df.isna().any().any() or not np.isfinite(df["revenue"]).all():
        raise ValueError("Missing or non-finite input values")
    if (df["revenue"] < 0).any():
        raise ValueError("Revenue cannot be negative")
    if (df["country"].astype(str).str.strip() == "ALL").any():
        raise ValueError("ALL is a reserved aggregate name")
    df["date"] = df["date"].dt.to_period("M").dt.to_timestamp()
    groups = df.groupby(["date", "country"], as_index=False)["revenue"].sum()
    totals = groups.groupby("date", as_index=False)["revenue"].sum()
    totals["country"] = "ALL"
    return pd.concat([groups, totals], ignore_index=True).sort_values(["date", "country"]).reset_index(drop=True)
