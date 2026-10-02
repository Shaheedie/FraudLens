from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .evaluation import choose_threshold, metrics_at_threshold
from .features import FEATURES, engineer_features


@dataclass
class CandidateResult:
    name: str
    model: object
    threshold: float
    validation: dict


def build_candidates(random_state: int = 42) -> dict:
    return {
        "logistic_regression": Pipeline(
            [
                ("scale", StandardScaler()),
                (
                    "model",
                    LogisticRegression(max_iter=2000, class_weight="balanced", random_state=random_state),
                ),
            ]
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=250,
            min_samples_leaf=2,
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=random_state,
        ),
        "hist_gradient_boosting": HistGradientBoostingClassifier(
            learning_rate=0.08,
            max_iter=220,
            max_leaf_nodes=31,
            l2_regularization=0.2,
            random_state=random_state,
        ),
    }


def split_chronologically(df, train_fraction=0.60, validation_fraction=0.20):
    ordered = df.sort_values("Time").reset_index(drop=True)
    n = len(ordered)
    a = int(n * train_fraction)
    b = int(n * (train_fraction + validation_fraction))
    return ordered.iloc[:a], ordered.iloc[a:b], ordered.iloc[b:]


def train_and_select(df, random_state=42):
    train, valid, test = split_chronologically(df)
    train_x = engineer_features(train)[FEATURES]
    valid_x = engineer_features(valid)[FEATURES]
    test_x = engineer_features(test)[FEATURES]
    y_train, y_valid, y_test = train["Class"], valid["Class"], test["Class"]

    results = []
    for name, model in build_candidates(random_state).items():
        fit_kwargs = {}
        if name == "hist_gradient_boosting":
            # Rare-class emphasis without synthetic observations.
            weights = np.where(y_train.to_numpy() == 1, 40.0, 1.0)
            fit_kwargs["sample_weight"] = weights
        model.fit(train_x, y_train, **fit_kwargs)
        scores = model.predict_proba(valid_x)[:, 1]
        threshold, _ = choose_threshold(y_valid, scores)
        results.append(
            CandidateResult(name, model, threshold, metrics_at_threshold(y_valid, scores, threshold))
        )

    selected = max(results, key=lambda r: (r.validation["pr_auc"], r.validation["f1"]))
    test_scores = selected.model.predict_proba(test_x)[:, 1]
    test_metrics = metrics_at_threshold(y_test, test_scores, selected.threshold)
    return selected, results, test_metrics, (train, valid, test)
