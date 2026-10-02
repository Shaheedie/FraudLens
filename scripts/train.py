from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from fraudlens.config import CLEAN_CSV, METADATA_PATH, MODEL_PATH, RANDOM_STATE, REPORT_DIR
from fraudlens.data import clean_dataset, load_dataset
from fraudlens.modeling import train_and_select


def main():
    df = load_dataset(CLEAN_CSV if CLEAN_CSV.exists() else ROOT / "data/raw/creditcard.csv")
    df, _ = clean_dataset(df)
    selected, results, test_metrics, splits = train_and_select(df, RANDOM_STATE)
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    artifact = {
        "model": selected.model,
        "threshold": selected.threshold,
        "features": [f"V{i}" for i in range(1, 29)] + ["Amount", "hour"],
    }
    joblib.dump(artifact, MODEL_PATH, compress=3)
    meta = {
        "selected_model": selected.name,
        "threshold": selected.threshold,
        "validation": selected.validation,
        "test": test_metrics,
        "split_rows": {"train": len(splits[0]), "validation": len(splits[1]), "test": len(splits[2])},
    }
    METADATA_PATH.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    pd.DataFrame([{"model": r.name, **r.validation} for r in results]).to_csv(
        REPORT_DIR / "model_comparison.csv", index=False
    )
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
