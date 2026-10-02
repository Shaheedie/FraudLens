from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd


def build_database(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(path) as conn:
        df.to_sql("transactions", conn, if_exists="replace", index=False)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_transactions_class ON transactions(Class)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_transactions_time ON transactions(Time)")
