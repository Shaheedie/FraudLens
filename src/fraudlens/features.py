from __future__ import annotations

import numpy as np
import pandas as pd

FEATURES = [*[f"V{i}" for i in range(1, 29)], "Amount", "hour"]


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["hour"] = (out["Time"] / 3600.0) % 24
    out["Amount"] = np.log1p(out["Amount"])
    return out
