from dataclasses import dataclass

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from app.seed.seed_data import INDICATORS


INDICATOR_COLUMNS = [indicator["code"] for indicator in INDICATORS]


@dataclass(slots=True)
class FeatureSet:
    frame: pd.DataFrame
    target: pd.Series
    numeric_columns: list[str]
    categorical_columns: list[str]


def build_feature_set(
    frame: pd.DataFrame,
    *,
    feature_mode: str,
    include_rule_scores: bool = False,
) -> FeatureSet:
    numeric_columns = INDICATOR_COLUMNS + ["age"]
    categorical_columns = ["sex"]

    if feature_mode == "hybrid":
        categorical_columns += [f"deviation_state_{code}" for code in INDICATOR_COLUMNS]
        numeric_columns += [
            column
            for column in frame.columns
            if column.startswith("pattern_")
        ]
    elif feature_mode != "raw_only":
        raise ValueError(f"Unsupported feature mode: {feature_mode}")

    if include_rule_scores:
        numeric_columns += [
            column
            for column in frame.columns
            if column.startswith("rule_score_")
        ]

    selected_columns = categorical_columns + numeric_columns
    feature_frame = frame[selected_columns].copy()
    for column in numeric_columns:
        feature_frame[column] = pd.to_numeric(feature_frame[column], errors="coerce")
    return FeatureSet(
        frame=feature_frame,
        target=frame["intended_label"].copy(),
        numeric_columns=numeric_columns,
        categorical_columns=categorical_columns,
    )


def build_preprocessor(feature_set: FeatureSet) -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                feature_set.categorical_columns,
            ),
            (
                "numeric",
                Pipeline(
                    steps=[
                        ("scaler", StandardScaler()),
                    ]
                ),
                feature_set.numeric_columns,
            ),
        ]
    )
