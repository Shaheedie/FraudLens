from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd

EXPECTED_COLUMNS = ["Time", *[f"V{i}" for i in range(1, 29)], "Amount", "Class"]


def load_dataset(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = [c for c in EXPECTED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing expected columns: {missing}")
    return df[EXPECTED_COLUMNS].copy()


def clean_dataset(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    before = len(df)
    duplicate_count = int(df.duplicated().sum())
    cleaned = df.drop_duplicates().reset_index(drop=True)
    if cleaned.isna().any().any():
        raise ValueError("Dataset contains missing values after loading.")
    if not set(cleaned["Class"].unique()).issubset({0, 1}):
        raise ValueError("Class must contain only 0 and 1.")
    if (cleaned["Amount"] < 0).any() or (cleaned["Time"] < 0).any():
        raise ValueError("Amount and Time must be non-negative.")
    info = {
        "rows_raw": before,
        "rows_clean": len(cleaned),
        "duplicates_removed": duplicate_count,
        "fraud_count": int(cleaned["Class"].sum()),
        "fraud_rate_pct": float(cleaned["Class"].mean() * 100),
    }
    return cleaned, info


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
