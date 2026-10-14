from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from .config import APP_ROOT, CAVEATS, DATA_PATH, MODEL_VERSION
from .preprocessing import build_feature_catalog, build_preprocessor, load_dataset

RANDOM_STATE = 42
SELECTION_STRATEGY = (
    "Highest validation F0.5, which weights precision more heavily to limit false "
    "alerts; precision and recall break ties. Thresholds are selected on validation "
    "data before the held-out test set is evaluated."
)


def dataset_fingerprint(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def choose_threshold(
    truth: pd.Series | np.ndarray,
    probabilities: np.ndarray,
) -> tuple[float, dict[str, float]]:
    """Choose an operating point on validation data, not the held-out test set."""
    precision, recall, thresholds = precision_recall_curve(truth, probabilities)
    if not len(thresholds):
        return 0.5, {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    candidates: list[tuple[float, float, float, float, float]] = []
    for index, threshold in enumerate(thresholds):
        p = float(precision[index])
        r = float(recall[index])
        f1 = (2 * p * r / (p + r)) if (p + r) else 0.0
        f0_5 = (1.25 * p * r / (0.25 * p + r)) if (0.25 * p + r) else 0.0
        candidates.append((f0_5, p, r, float(threshold), f1))
    best_f0_5, best_precision, best_recall, best_threshold, best_f1 = max(
        candidates
    )
    return best_threshold, {
        "precision": best_precision,
        "recall": best_recall,
        "f1": best_f1,
        "f0_5": best_f0_5,
    }


def _build_models(positive_weight: float) -> dict[str, Any]:
    return {
        "Logistic Regression": LogisticRegression(
            class_weight="balanced",
            max_iter=1500,
            solver="liblinear",
            random_state=RANDOM_STATE,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=240,
            max_features="sqrt",
            min_samples_leaf=2,
            class_weight="balanced_subsample",
            n_jobs=2,
            random_state=RANDOM_STATE,
        ),
        "XGBoost": XGBClassifier(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
            min_child_weight=2,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_lambda=1.5,
            scale_pos_weight=positive_weight,
            objective="binary:logistic",
            eval_metric="aucpr",
            tree_method="hist",
            n_jobs=2,
            random_state=RANDOM_STATE,
            verbosity=0,
        ),
    }


def _metric_row(
    name: str,
    truth: pd.Series,
    probabilities: np.ndarray,
    threshold: float,
) -> dict[str, Any]:
    predictions = (probabilities >= threshold).astype("int8")
    tn, fp, fn, tp = confusion_matrix(truth, predictions, labels=[0, 1]).ravel()
    return {
        "model": name,
        "precision": round(
            float(precision_score(truth, predictions, zero_division=0)), 4
        ),
        "recall": round(float(recall_score(truth, predictions, zero_division=0)), 4),
        "f1": round(float(f1_score(truth, predictions, zero_division=0)), 4),
        "rocAuc": round(float(roc_auc_score(truth, probabilities)), 4),
        "prAuc": round(float(average_precision_score(truth, probabilities)), 4),
        "threshold": round(float(threshold), 4),
        "confusionMatrix": {
            "trueNegative": int(tn),
            "falsePositive": int(fp),
            "falseNegative": int(fn),
            "truePositive": int(tp),
        },
    }


def train_model_bundle(csv_path: str | Path = DATA_PATH) -> dict[str, Any]:
    """Train and compare weighted classifiers with stratified train/validation/test sets."""
    features, target, numeric_columns, categorical_columns = load_dataset(csv_path)
    x_train_all, x_test, y_train_all, y_test = train_test_split(
        features,
        target,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=target,
    )
    x_fit, x_validation, y_fit, y_validation = train_test_split(
        x_train_all,
        y_train_all,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=y_train_all,
    )

    positive_count = int(y_fit.sum())
    negative_count = int(len(y_fit) - positive_count)
    if positive_count == 0 or negative_count == 0:
        raise ValueError("The training target must contain both fraud classes.")
    candidates = _build_models(negative_count / positive_count)
    fitted_models: dict[str, Pipeline] = {}
    thresholds: dict[str, float] = {}
    validation_selection: dict[str, dict[str, float]] = {}
    model_metrics: list[dict[str, Any]] = []

    for name, classifier in candidates.items():
        pipeline = Pipeline(
            steps=[
                ("preprocess", build_preprocessor(numeric_columns, categorical_columns)),
                ("classifier", classifier),
            ]
        )
        pipeline.fit(x_fit, y_fit)
        validation_probabilities = pipeline.predict_proba(x_validation)[:, 1]
        threshold, validation_metrics = choose_threshold(
            y_validation, validation_probabilities
        )
        test_probabilities = pipeline.predict_proba(x_test)[:, 1]

        fitted_models[name] = pipeline
        thresholds[name] = threshold
        validation_selection[name] = validation_metrics
        model_metrics.append(
            _metric_row(name, y_test, test_probabilities, threshold)
        )

    selected_model = max(
        validation_selection,
        key=lambda name: (
            validation_selection[name]["f0_5"],
            validation_selection[name]["precision"],
            validation_selection[name]["recall"],
        ),
    )
    model_metrics.sort(
        key=lambda row: (
            row["model"] != selected_model,
            -validation_selection[row["model"]]["f0_5"],
            -row["precision"],
        )
    )
    return {
        "modelVersion": MODEL_VERSION,
        "datasetFingerprint": dataset_fingerprint(csv_path),
        "datasetRows": int(len(features)),
        "fraudCases": int(target.sum()),
        "featureNames": list(features.columns),
        "numericColumns": numeric_columns,
        "categoricalColumns": categorical_columns,
        "featureCatalog": build_feature_catalog(features),
        "models": fitted_models,
        "thresholds": thresholds,
        "selectedModel": selected_model,
        "selectionStrategy": SELECTION_STRATEGY,
        "testRecords": int(len(x_test)),
        "modelMetrics": model_metrics,
        "caveats": list(CAVEATS),
    }


if __name__ == "__main__":
    from .model_manager import get_model_bundle

    bundle = get_model_bundle(force_retrain=True)
    print(
        pd.Series(
            {
                "datasetRows": bundle["datasetRows"],
                "fraudCases": bundle["fraudCases"],
                "selectedModel": bundle["selectedModel"],
                "decisionThreshold": bundle["thresholds"][bundle["selectedModel"]],
            }
        ).to_json(indent=2)
    )
    print(pd.DataFrame(bundle["modelMetrics"]).to_string(index=False))
