from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from fraudlens.config import MODEL_PATH
from fraudlens.features import FEATURES, engineer_features


def main():
    p = argparse.ArgumentParser()
    p.add_argument("json_file", type=Path)
    a = p.parse_args()
    artifact = joblib.load(MODEL_PATH)
    row = json.loads(a.json_file.read_text())
    frame = pd.DataFrame([row])
    x = engineer_features(frame)[FEATURES]
    score = float(artifact["model"].predict_proba(x)[0, 1])
    threshold = float(artifact["threshold"])
    print(
        json.dumps(
            {
                "fraud_probability": score,
                "risk_percent": round(score * 100, 2),
                "flagged": score >= threshold,
                "threshold": threshold,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
