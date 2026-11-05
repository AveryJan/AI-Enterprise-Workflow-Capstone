import pandas as pd
import pytest
from src.data import make_demo_data, ingest


def test_ingestion_and_aggregate(tmp_path):
    path = make_demo_data(tmp_path / "sample.csv")
    df = ingest(path)
    month = df["date"].min()
    this_month = df.loc[df["date"] == month]
    total = this_month.loc[this_month["country"] == "ALL", "revenue"].iloc[0]
    components = this_month.loc[this_month["country"] != "ALL", "revenue"].sum()
    assert total == pytest.approx(components)


def test_bad_input(tmp_path):
    path = tmp_path / "bad.csv"
    pd.DataFrame({"country": ["Canada"], "date": ["2022-01-01"]}).to_csv(path, index=False)
    with pytest.raises(ValueError):
        ingest(path)
