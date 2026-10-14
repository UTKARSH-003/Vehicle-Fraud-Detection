from __future__ import annotations

import fcntl
import os
import tempfile
import threading
from pathlib import Path
from typing import Any

import joblib

from .config import DATA_PATH, MODEL_DIR, MODEL_PATH, MODEL_VERSION
from .training import dataset_fingerprint, train_model_bundle

_memory_lock = threading.RLock()
_cached_bundle: dict[str, Any] | None = None


def get_model_bundle(
    force_retrain: bool = False,
    csv_path: str | Path = DATA_PATH,
) -> dict[str, Any]:
    """Load the current data-matched model bundle or train it once if absent."""
    global _cached_bundle
    path = Path(csv_path)
    current_fingerprint = dataset_fingerprint(path)

    with _memory_lock:
        if (
            not force_retrain
            and _cached_bundle is not None
            and _cached_bundle.get("datasetFingerprint") == current_fingerprint
            and _cached_bundle.get("modelVersion") == MODEL_VERSION
        ):
            return _cached_bundle

        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        lock_path = MODEL_PATH.with_suffix(".lock")
        with lock_path.open("w", encoding="utf-8") as lock_file:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
            if MODEL_PATH.is_file() and not force_retrain:
                try:
                    saved = joblib.load(MODEL_PATH)
                    if (
                        saved.get("datasetFingerprint") == current_fingerprint
                        and saved.get("modelVersion") == MODEL_VERSION
                    ):
                        _cached_bundle = saved
                        return saved
                except (EOFError, OSError, ValueError, AttributeError):
                    # A partial local artifact is rebuilt rather than served.
                    pass

            bundle = train_model_bundle(path)
            fd, temporary_name = tempfile.mkstemp(
                prefix="model-bundle-", suffix=".joblib", dir=MODEL_DIR
            )
            os.close(fd)
            try:
                joblib.dump(bundle, temporary_name, compress=3)
                os.replace(temporary_name, MODEL_PATH)
            finally:
                if os.path.exists(temporary_name):
                    os.remove(temporary_name)
            _cached_bundle = bundle
            return bundle
