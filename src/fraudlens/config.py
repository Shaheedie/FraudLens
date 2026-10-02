from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports"
FIGURE_DIR = REPORT_DIR / "figures"
RAW_CSV = DATA_RAW / "creditcard.csv"
CLEAN_CSV = DATA_PROCESSED / "creditcard_clean.csv"
DB_PATH = DATA_PROCESSED / "fraudlens.db"
MODEL_PATH = MODEL_DIR / "fraudlens_model.joblib"
METADATA_PATH = MODEL_DIR / "fraudlens_model.json"
RANDOM_STATE = 42
