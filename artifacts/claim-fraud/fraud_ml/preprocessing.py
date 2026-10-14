from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .config import (
    CATEGORY_ORDER,
    EXCLUDED_COLUMNS,
    FEATURE_GROUPS,
    FEATURE_LABELS,
    TARGET_COLUMN,
)


def load_dataset(
    csv_path: str | Path,
) -> tuple[pd.DataFrame, pd.Series, list[str], list[str]]:
    """Read and normalize the source CSV, returning features and a binary target."""
    path = Path(csv_path)
    if not path.is_file():
        raise FileNotFoundError(f"Training dataset was not found: {path}")

    frame = pd.read_csv(
        path,
        na_values=["?", "", "NA", "N/A", "null", "None"],
        keep_default_na=True,
    )
    frame.columns = [str(column).strip() for column in frame.columns]
    if TARGET_COLUMN not in frame.columns:
        raise ValueError(
            f"Dataset must include the target column {TARGET_COLUMN!r}."
        )

    target_values = frame[TARGET_COLUMN].astype("string").str.strip().str.lower()
    target = target_values.map(
        {
            "0": 0,
            "1": 1,
            "no": 0,
            "yes": 1,
            "false": 0,
            "true": 1,
        }
    )
    keep_rows = target.notna()
    frame = frame.loc[keep_rows].copy()
    target = target.loc[keep_rows].astype("int8")

    for column in frame.columns:
        if column == TARGET_COLUMN:
            continue
        if pd.api.types.is_object_dtype(frame[column]) or isinstance(
            frame[column].dtype, pd.StringDtype
        ):
            values = frame[column].astype("string").str.strip()
            frame[column] = values.replace({"": pd.NA, "?": pd.NA})

    feature_columns = [
        column for column in frame.columns if column not in EXCLUDED_COLUMNS
    ]
    if not feature_columns:
        raise ValueError("No eligible model features remain after exclusions.")

    features = frame.loc[:, feature_columns].copy()
    numeric_columns = list(
        features.select_dtypes(include=["number", "bool"]).columns
    )
    categorical_columns = [
        column for column in features.columns if column not in numeric_columns
    ]

    for column in numeric_columns:
        features[column] = pd.to_numeric(features[column], errors="coerce")
    for column in categorical_columns:
        features[column] = features[column].astype("object")
        features[column] = features[column].where(features[column].notna(), np.nan)

    return features, target, numeric_columns, categorical_columns


def build_preprocessor(
    numeric_columns: list[str],
    categorical_columns: list[str],
) -> ColumnTransformer:
    """Impute missing values, scale numbers, and one-hot encode categories."""
    numeric = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
            ("scaler", StandardScaler()),
        ]
    )
    categorical = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="ignore",
                    min_frequency=2,
                    sparse_output=True,
                ),
            ),
        ]
    )
    transformers: list[tuple[str, Any, list[str]]] = []
    if numeric_columns:
        transformers.append(("numeric", numeric, numeric_columns))
    if categorical_columns:
        transformers.append(("categorical", categorical, categorical_columns))
    return ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        sparse_threshold=0.3,
        verbose_feature_names_out=False,
    )


def build_feature_catalog(features: pd.DataFrame) -> list[dict[str, Any]]:
    """Create the form schema from the model's allowed training columns."""
    catalog: list[dict[str, Any]] = []
    for name in features.columns:
        series = features[name]
        label = FEATURE_LABELS.get(name, name.replace("_", " "))
        group = next(
            (
                group_name
                for group_name, columns in FEATURE_GROUPS.items()
                if name in columns
            ),
            "Claim details",
        )

        if pd.api.types.is_numeric_dtype(series):
            values = pd.to_numeric(series, errors="coerce").dropna()
            minimum = float(values.min()) if not values.empty else None
            maximum = float(values.max()) if not values.empty else None
            is_integer = bool(
                not values.empty
                and np.all(np.isclose(values.to_numpy(), np.round(values.to_numpy())))
            )
            step = 1.0 if is_integer else 0.1
            catalog.append(
                {
                    "name": name,
                    "label": label,
                    "group": group,
                    "kind": "numeric",
                    "options": [],
                    "minimum": minimum,
                    "maximum": maximum,
                    "step": step,
                }
            )
            continue

        values = [
            str(value)
            for value in pd.unique(series.dropna())
            if str(value).strip()
        ]
        preferred_order = CATEGORY_ORDER.get(name, [])
        ordered = [value for value in preferred_order if value in values]
        ordered.extend(sorted(value for value in values if value not in ordered))
        catalog.append(
            {
                "name": name,
                "label": label,
                "group": group,
                "kind": "categorical",
                "options": ordered,
                "minimum": None,
                "maximum": None,
                "step": None,
            }
        )
    return catalog
