from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from fraudlens.config import CLEAN_CSV, RAW_CSV
from fraudlens.data import clean_dataset, load_dataset


def main():
    if not RAW_CSV.exists():
        raise SystemExit(
            "Missing data/raw/creditcard.csv. Download the ULB dataset from Kaggle and place it there."
        )
    df = load_dataset(RAW_CSV)
    clean, info = clean_dataset(df)
    CLEAN_CSV.parent.mkdir(parents=True, exist_ok=True)
    clean.to_csv(CLEAN_CSV, index=False)
    print(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()
