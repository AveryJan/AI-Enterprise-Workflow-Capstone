"""Opt-in prediction logging; test paths can be redirected outside production."""
from pathlib import Path
import csv
import json
import threading
from datetime import datetime, timezone

_lock = threading.Lock()


def record_prediction(log_path, country, date, prediction):
    path = Path(log_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with _lock:
        existing = path.exists() and path.stat().st_size > 0
        with path.open("a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["timestamp_utc", "country", "date", "prediction"])
            if not existing:
                writer.writeheader()
            writer.writerow({"timestamp_utc": datetime.now(timezone.utc).isoformat(),
                             "country": country, "date": date, "prediction": prediction})


def performance_report(metric_path, log_path):
    metric_path, log_path = Path(metric_path), Path(log_path)
    if not metric_path.exists():
        return {"model_metrics_available": False, "predictions_logged": 0}
    metrics = json.loads(metric_path.read_text())
    if log_path.exists():
        with log_path.open(newline="") as f:
            count = max(sum(1 for _ in csv.reader(f)) - 1, 0)
    else:
        count = 0
    return {"model_metrics_available": True, "selected_model": metrics["selected_model"],
            "validation": metrics["validation"], "predictions_logged": count}
