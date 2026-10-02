import numpy as np
import pandas as pd

from fraudlens.data import clean_dataset
from fraudlens.evaluation import choose_threshold, metrics_at_threshold
from fraudlens.features import engineer_features


def test_clean_removes_duplicates():
    row = {"Time": 0, **{f"V{i}": 0.0 for i in range(1, 29)}, "Amount": 10, "Class": 0}
    cleaned, info = clean_dataset(pd.DataFrame([row, row]))
    assert len(cleaned) == 1
    assert info["duplicates_removed"] == 1


def test_feature_engineering():
    df = pd.DataFrame([{"Time": 7200, "Amount": 99.0}])
    out = engineer_features(df)
    assert out.loc[0, "hour"] == 2
    assert out.loc[0, "Amount"] > 0


def test_metrics_and_threshold():
    y = np.array([0, 0, 1, 1])
    scores = np.array([0.1, 0.2, 0.6, 0.9])
    m = metrics_at_threshold(y, scores, 0.5)
    assert m["f1"] == 1.0
    t, frame = choose_threshold(y, scores)
    assert 0 < t < 1
    assert not frame.empty
