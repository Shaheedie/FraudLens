from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from fraudlens.config import CLEAN_CSV, DB_PATH
from fraudlens.data import load_dataset
from fraudlens.database import build_database

if __name__ == "__main__":
    build_database(load_dataset(CLEAN_CSV), DB_PATH)
    print(f"Database -> {DB_PATH}")
