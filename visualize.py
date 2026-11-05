"""Reproducible EDA and baseline comparison graphics."""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from src.data import ingest


def main():
    out = Path("figures")
    out.mkdir(exist_ok=True)
    df = ingest("data/demo_revenue.csv")
    fig, ax = plt.subplots(figsize=(9, 5))
    for country, part in df.groupby("country"):
        ax.plot(part["date"], part["revenue"], label=country)
    ax.set(title="Exploratory analysis: revenue by country", xlabel="Month", ylabel="Synthetic revenue")
    ax.legend()
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(out / "eda_revenue_by_country.png", dpi=150)
    plt.close(fig)

    preds = pd.read_csv("artifacts/validation_predictions.csv")
    mae = preds.assign(error=(preds["predicted"] - preds["actual"]).abs()).groupby("model")["error"].mean().sort_values()
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(mae.index, mae.values)
    ax.set(title="Validation comparison versus baseline", ylabel="Mean absolute error (lower is better)")
    fig.tight_layout()
    fig.savefig(out / "model_vs_baseline.png", dpi=150)
    plt.close(fig)

if __name__ == "__main__":
    main()
