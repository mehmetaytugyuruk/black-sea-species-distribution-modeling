"""Model training utilities."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import train_test_split


@dataclass
class TrainedModelResult:
    """Container for an evaluated species/model pair."""

    species: str
    model_name: str
    model: Any
    final_model: Any
    feature_columns: list[str]
    y_test: pd.Series
    y_score: np.ndarray
    y_pred: np.ndarray
    n_train: int
    n_test: int
    n_total: int


def train_species_models(
    training_data: pd.DataFrame,
    feature_columns: list[str],
    target_column: str,
    model_names: list[str],
    test_size: float,
    random_seed: int,
    logger: logging.Logger | None = None,
) -> list[TrainedModelResult]:
    """Train configured classifiers separately for each species."""

    log = logger or logging.getLogger(__name__)
    results: list[TrainedModelResult] = []

    required_columns = feature_columns + [target_column, "species"]
    model_data = training_data.dropna(subset=required_columns).copy()
    model_data[target_column] = model_data[target_column].astype(int)

    for species, group in model_data.groupby("species"):
        class_counts = group[target_column].value_counts()
        if class_counts.get(0, 0) < 2 or class_counts.get(1, 0) < 2:
            log.warning("Skipping %s because it does not have enough presence/background records", species)
            continue

        X = group[feature_columns]
        y = group[target_column]
        stratify = y if class_counts.min() >= 2 else None
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=random_seed,
            stratify=stratify,
        )

        for model_name in model_names:
            log.info("Training %s for %s", model_name, species)
            model = make_model(model_name, random_seed=random_seed)
            model.fit(X_train, y_train)
            y_score = _predict_score(model, X_test)
            y_pred = (y_score >= 0.5).astype(int)

            final_model = make_model(model_name, random_seed=random_seed)
            final_model.fit(X, y)

            results.append(
                TrainedModelResult(
                    species=str(species),
                    model_name=model_name,
                    model=model,
                    final_model=final_model,
                    feature_columns=feature_columns,
                    y_test=y_test,
                    y_score=y_score,
                    y_pred=y_pred,
                    n_train=len(X_train),
                    n_test=len(X_test),
                    n_total=len(X),
                )
            )

    return results


def make_model(model_name: str, random_seed: int) -> Any:
    """Create a classifier by config name."""

    if model_name == "random_forest":
        return RandomForestClassifier(
            n_estimators=200,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=random_seed,
            n_jobs=-1,
        )
    if model_name == "gradient_boosting":
        return GradientBoostingClassifier(random_state=random_seed)
    raise ValueError(f"Unsupported model name: {model_name}")


def _predict_score(model: Any, features: pd.DataFrame) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return model.predict_proba(features)[:, 1]
    if hasattr(model, "decision_function"):
        scores = model.decision_function(features)
        return 1 / (1 + np.exp(-scores))
    return model.predict(features).astype(float)

