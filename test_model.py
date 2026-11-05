import joblib
import pytest
from src.data import make_demo_data, ingest
from src.model import predict, train


def test_training_compares_multiple_models_and_predicts(tmp_path):
    df = ingest(make_demo_data(tmp_path / "data.csv"))
    metrics = train(df, tmp_path / "models")
    assert set(metrics["validation"]) == {"baseline", "ridge", "random_forest"}
    assert all(v["mae"] >= 0 for v in metrics["validation"].values())
    bundle = joblib.load(tmp_path / "models" / "model.joblib")
    assert predict(bundle, "Canada", "2025-01-01") >= 0
    assert predict(bundle, "ALL", "2025-01-01") >= 0
    with pytest.raises(ValueError):
        predict(bundle, "Neverland", "2025-01-01")
