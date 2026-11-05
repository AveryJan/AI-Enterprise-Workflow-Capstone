import json
from threading import Thread
from urllib.request import urlopen
from urllib.error import HTTPError
from src.data import ingest, make_demo_data
from src.model import train
from src.api import create_server


def test_api_country_and_all_and_validation(tmp_path):
    train(ingest(make_demo_data(tmp_path / "source.csv")), tmp_path / "models")
    server = create_server(model_path=tmp_path / "models" / "model.joblib",
                           metrics_path=tmp_path / "models" / "metrics.json",
                           log_path=tmp_path / "test_logs" / "predictions.csv")
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_port}"
    def get(route):
        with urlopen(base + route, timeout=5) as response:
            return response.status, json.loads(response.read())
    try:
        assert get("/health")[0] == 200
        code, country = get("/predict?country=Canada&date=2025-01-01")
        code2, all_countries = get("/predict?country=ALL&date=2025-01-01")
        assert code == code2 == 200
        assert country["country"] == "Canada" and all_countries["country"] == "ALL"
        assert isinstance(country["prediction"], float)
        for invalid in ["/predict?country=BAD&date=2025-01-01", "/predict?country=Canada"]:
            try:
                get(invalid)
                assert False, "expected 400"
            except HTTPError as exc:
                assert exc.code == 400
        assert get("/metrics")[1]["predictions_logged"] == 2
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
