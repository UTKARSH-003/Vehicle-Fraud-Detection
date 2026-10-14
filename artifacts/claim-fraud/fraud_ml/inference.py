from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .model_manager import get_model_bundle


def get_overview() -> dict[str, Any]:
    bundle = get_model_bundle()
    return {
        "status": "ready",
        "datasetRows": bundle["datasetRows"],
        "fraudCases": bundle["fraudCases"],
        "fraudRate": round(bundle["fraudCases"] / bundle["datasetRows"], 4),
        "featureCount": len(bundle["featureNames"]),
        "selectedModel": bundle["selectedModel"],
        "selectionStrategy": bundle["selectionStrategy"],
        "decisionThreshold": round(
            float(bundle["thresholds"][bundle["selectedModel"]]), 4
        ),
        "caveats": bundle["caveats"],
    }


def get_metrics() -> dict[str, Any]:
    bundle = get_model_bundle()
    return {
        "selectedModel": bundle["selectedModel"],
        "testRecords": bundle["testRecords"],
        "selectionStrategy": bundle["selectionStrategy"],
        "models": bundle["modelMetrics"],
    }


def get_feature_catalog() -> dict[str, Any]:
    bundle = get_model_bundle()
    return {
        "featureCount": len(bundle["featureCatalog"]),
        "features": bundle["featureCatalog"],
    }


def predict_claim(
    raw_features: Any,
    requested_threshold: Any = None,
) -> dict[str, Any]:
    bundle = get_model_bundle()
    if not isinstance(raw_features, dict):
        raise ValueError("features must be an object of feature names and values.")

    allowed = set(bundle["featureNames"])
    received = set(raw_features)
    unknown = sorted(received - allowed)
    missing = sorted(allowed - received)
    if unknown:
        raise ValueError(f"Unknown claim fields: {', '.join(unknown)}.")
    if missing:
        raise ValueError(f"Missing claim fields: {', '.join(missing)}.")

    numeric = set(bundle["numericColumns"])
    row: dict[str, Any] = {}
    for name in bundle["featureNames"]:
        value = raw_features[name]
        if name in numeric:
            try:
                number = float(value)
            except (TypeError, ValueError) as error:
                raise ValueError(f"{name} must be a number.") from error
            if not np.isfinite(number):
                raise ValueError(f"{name} must be a finite number.")
            row[name] = number
        else:
            text = str(value).strip()
            if not text:
                raise ValueError(f"{name} cannot be empty.")
            row[name] = text

    model_name = bundle["selectedModel"]
    default_threshold = float(bundle["thresholds"][model_name])
    if requested_threshold is None:
        threshold = default_threshold
    else:
        try:
            threshold = float(requested_threshold)
        except (TypeError, ValueError) as error:
            raise ValueError("threshold must be a number between 0 and 1.") from error
        if not np.isfinite(threshold) or not 0 <= threshold <= 1:
            raise ValueError("threshold must be a number between 0 and 1.")

    model = bundle["models"][model_name]
    probability = float(
        model.predict_proba(pd.DataFrame([row], columns=bundle["featureNames"]))[0, 1]
    )
    review_floor = min(0.5, max(0.05, threshold * 0.6))
    if probability >= threshold:
        risk_level = "high"
        recommendation = (
            "Prioritize an evidence-based investigator review. This score is not a "
            "finding of fraud."
        )
    elif probability >= review_floor:
        risk_level = "review"
        recommendation = (
            "Consider a focused manual review and verify supporting documentation "
            "before making a claim decision."
        )
    else:
        risk_level = "low"
        recommendation = (
            "No elevated signal at the selected threshold; continue the standard "
            "claims process."
        )

    return {
        "fraudRiskScore": round(probability, 4),
        "decisionThreshold": round(threshold, 4),
        "riskLevel": risk_level,
        "recommendation": recommendation,
        "model": model_name,
        "limitations": bundle["caveats"],
    }
