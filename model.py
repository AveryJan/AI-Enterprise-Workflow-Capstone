"""Chronological-validation country-level revenue forecasting models."""
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

FEATURES = ["country", "month", "time_index"]


def features(df, origin):
    dates = pd.to_datetime(df["date"])
    return pd.DataFrame({"country": df["country"].astype(str).to_numpy(),
                         "month": dates.dt.month.to_numpy(),
                         "time_index": ((dates.dt.year - origin.year) * 12 + dates.dt.month - origin.month).to_numpy()})


def candidates():
    pre = ColumnTransformer([("country", OneHotEncoder(handle_unknown="ignore"), ["country"]),
                             ("numeric", "passthrough", ["month", "time_index"])])
    return {"baseline": DummyRegressor(strategy="mean"),
            "ridge": Pipeline([("pre", pre), ("model", Ridge(alpha=1.0))]),
            "random_forest": Pipeline([("pre", pre), ("model", RandomForestRegressor(n_estimators=100, random_state=42, min_samples_leaf=2))])}


def train(df, output_dir):
    """Hold out the newest 20% of unique months; never random-shuffle time-series."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    dates = sorted(pd.to_datetime(df["date"]).unique())
    if len(dates) < 10:
        raise ValueError("At least 10 unique months required")
    cutoff = dates[int(len(dates) * 0.8)]
    train_mask = pd.to_datetime(df["date"]) < cutoff
    origin = pd.Timestamp(dates[0])
    X = features(df, origin)
    y = df["revenue"].to_numpy()
    metrics = {}
    predictions = []
    for name, estimator in candidates().items():
        estimator.fit(X.loc[train_mask], y[train_mask])
        yhat = estimator.predict(X.loc[~train_mask])
        metrics[name] = {"mae": round(float(mean_absolute_error(y[~train_mask], yhat)), 3),
                         "rmse": round(float(np.sqrt(mean_squared_error(y[~train_mask], yhat))), 3)}
        for actual, predicted in zip(y[~train_mask], yhat):
            predictions.append({"model": name, "actual": float(actual), "predicted": float(predicted)})
    winner = min((k for k in metrics if k != "baseline"), key=lambda name: metrics[name]["mae"])
    best = candidates()[winner]
    best.fit(X, y)
    joblib.dump({"model": best, "origin": origin, "countries": sorted(df["country"].unique()),
                 "model_name": winner}, output_dir / "model.joblib")
    summary = {"selected_model": winner, "validation": metrics,
               "validation_start": str(pd.Timestamp(cutoff).date()),
               "validation_end": str(pd.Timestamp(dates[-1]).date()),
               "note": "Synthetic demonstration dataset; illustrative metrics, not production validation."}
    (output_dir / "metrics.json").write_text(json.dumps(summary, indent=2))
    pd.DataFrame(predictions).to_csv(output_dir / "validation_predictions.csv", index=False)
    return summary


def predict(bundle, country, date):
    if country not in bundle["countries"]:
        raise ValueError("Unknown country. Valid values: " + ", ".join(bundle["countries"]))
    date = pd.Timestamp(date)
    if date.day != 1:
        raise ValueError("Date must be first day of month (YYYY-MM-01)")
    frame = pd.DataFrame({"country": [country], "date": [date]})
    value = float(bundle["model"].predict(features(frame, bundle["origin"]))[0])
    return round(max(value, 0), 2)
