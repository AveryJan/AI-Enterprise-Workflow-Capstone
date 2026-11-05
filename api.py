"""Dependency-light HTTP prediction API with injectable test artifact and logging paths."""
import json
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit, parse_qs
import joblib
from src.model import predict
from src.monitoring import record_prediction, performance_report


def create_server(host="127.0.0.1", port=0, model_path=None, metrics_path=None, log_path=None):
    model_path = Path(model_path or os.getenv("MODEL_PATH", "artifacts/model.joblib"))
    metrics_path = Path(metrics_path or os.getenv("METRICS_PATH", "artifacts/metrics.json"))
    log_path = Path(log_path or os.getenv("PREDICTION_LOG", "logs/predictions.csv"))
    bundle = joblib.load(model_path) if model_path.exists() else None

    class Handler(BaseHTTPRequestHandler):
        def respond(self, status, payload):
            data = json.dumps(payload).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            path = urlsplit(self.path)
            if path.path == "/health":
                self.respond(200 if bundle else 503, {"status": "ok" if bundle else "model_unavailable"})
            elif path.path == "/metrics":
                self.respond(200, performance_report(metrics_path, log_path))
            elif path.path == "/predict":
                if bundle is None:
                    self.respond(503, {"error": "Model not trained. Run training first."})
                    return
                args = parse_qs(path.query)
                country = args.get("country", ["ALL"])[0]
                date = args.get("date", [""])[0]
                if not date:
                    self.respond(400, {"error": "date required (YYYY-MM-01)"})
                    return
                try:
                    value = predict(bundle, country, date)
                except (ValueError, TypeError, OverflowError) as exc:
                    self.respond(400, {"error": str(exc)})
                    return
                record_prediction(log_path, country, date, value)
                self.respond(200, {"country": country, "date": date, "prediction": value,
                                   "model": bundle["model_name"], "units": "synthetic_revenue"})
            else:
                self.respond(404, {"error": "route not found"})

        def log_message(self, format, *args):
            pass

    return ThreadingHTTPServer((host, port), Handler)


if __name__ == "__main__":
    server = create_server(host="0.0.0.0", port=int(os.getenv("PORT", "5000")))
    print(f"Serving predictions on port {server.server_port}", flush=True)
    server.serve_forever()
