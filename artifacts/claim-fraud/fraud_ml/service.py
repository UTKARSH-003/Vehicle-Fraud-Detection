from __future__ import annotations

import logging
import os
import sys

from .model_manager import get_model_bundle
from .web import app


def main() -> None:
    logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"))
    logging.getLogger(__name__).info("Preparing vehicle-claim fraud models.")
    bundle = get_model_bundle()
    logging.getLogger(__name__).info(
        "Model ready: selected=%s rows=%s fraud_cases=%s",
        bundle["selectedModel"],
        bundle["datasetRows"],
        bundle["fraudCases"],
    )
    port = int(os.environ.get("FRAUD_SERVICE_PORT", os.environ.get("PORT", "5000")))
    host = os.environ.get("FRAUD_SERVICE_HOST", "127.0.0.1")
    os.execv(
        sys.executable,
        [
            sys.executable,
            "-m",
            "gunicorn",
            "--bind",
            f"{host}:{port}",
            "--workers",
            os.environ.get("FRAUD_WORKERS", "1"),
            "--threads",
            os.environ.get("FRAUD_THREADS", "8"),
            "--timeout",
            "120",
            "--access-logfile",
            "-",
            "--error-logfile",
            "-",
            "fraud_ml.web:app",
        ],
    )


if __name__ == "__main__":
    main()
