from __future__ import annotations

import os
from pathlib import Path

from flask import Flask, jsonify, request, send_from_directory

from .config import APP_ROOT
from .inference import get_feature_catalog, get_metrics, get_overview, predict_claim

STATIC_ROOT = APP_ROOT / "dist" / "public"
app = Flask(__name__, static_folder=None)


@app.after_request
def add_security_headers(response):
    response.headers.setdefault("X-Content-Type-Options", "nosniff")
    response.headers.setdefault("Referrer-Policy", "strict-origin-when-cross-origin")
    response.headers.setdefault("Cache-Control", "no-store")
    return response


@app.get("/api/fraud/healthz")
def health():
    return jsonify({"status": "ok"})


@app.get("/api/fraud/overview")
def overview():
    return jsonify(get_overview())


@app.get("/api/fraud/metrics")
def metrics():
    return jsonify(get_metrics())


@app.get("/api/fraud/features")
def feature_catalog():
    return jsonify(get_feature_catalog())


@app.post("/api/fraud/predict")
def prediction():
    body = request.get_json(silent=True)
    if not isinstance(body, dict):
        return jsonify({"error": "Request body must be a JSON object."}), 400
    try:
        result = predict_claim(body.get("features"), body.get("threshold"))
    except ValueError as error:
        return jsonify({"error": str(error)}), 400
    return jsonify(result)


@app.errorhandler(404)
def not_found(_error):
    return jsonify({"error": "Endpoint not found."}), 404


@app.route("/", defaults={"requested_path": ""})
@app.route("/<path:requested_path>")
def frontend(requested_path: str):
    if requested_path == "api" or requested_path.startswith("api/"):
        return jsonify({"error": "Endpoint not found."}), 404
    if not STATIC_ROOT.is_dir():
        return (
            "Frontend build is missing. Run the build command in README.md first.",
            503,
        )
    candidate = (STATIC_ROOT / requested_path).resolve()
    if candidate != STATIC_ROOT and STATIC_ROOT not in candidate.parents:
        return jsonify({"error": "Not found."}), 404
    if requested_path and candidate.is_file():
        return send_from_directory(STATIC_ROOT, requested_path)
    return send_from_directory(STATIC_ROOT, "index.html")


if __name__ == "__main__":
    port = int(os.environ.get("FRAUD_SERVICE_PORT", os.environ.get("PORT", "5000")))
    host = os.environ.get("FRAUD_SERVICE_HOST", "127.0.0.1")
    app.run(host=host, port=port, debug=False, threaded=True)
