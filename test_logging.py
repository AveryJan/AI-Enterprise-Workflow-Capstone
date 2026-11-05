from src.monitoring import record_prediction, performance_report


def test_logging_isolated_from_production(tmp_path):
    log = tmp_path / "isolated" / "predictions.csv"
    record_prediction(log, "Canada", "2025-01-01", 100.25)
    record_prediction(log, "ALL", "2025-01-01", 300.25)
    report = performance_report(tmp_path / "missing.json", log)
    assert report["predictions_logged"] == 0  # no metrics available
    lines = log.read_text().splitlines()
    assert len(lines) == 3
    assert "Canada" in lines[1]
