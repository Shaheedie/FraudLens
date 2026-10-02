# FraudLens — Cost-Aware Credit Card Fraud Detection

FraudLens is an end-to-end machine-learning portfolio project for **rare-event fraud detection**. It is designed to demonstrate the parts of fraud modeling that ordinary accuracy-based classifiers miss: extreme class imbalance, chronological evaluation, precision-recall analysis, cost-aware decision thresholds, reproducibility, SQL analytics, and responsible model use.

## Why this project

The public ULB/Worldline benchmark contains **284,807 transactions and only 492 frauds (~0.172%)**. A classifier that predicts “legitimate” for every transaction would therefore look extremely accurate while detecting no fraud. FraudLens treats **PR-AUC, precision, recall, F1 and operational threshold costs** as first-class metrics.

## Architecture

```text
creditcard.csv
    ↓
data validation + duplicate removal
    ↓
SQLite analytics + feature engineering
    ↓
chronological 60 / 20 / 20 split
    ↓
Logistic Regression ─┐
Random Forest ────────┼→ validation PR-AUC → selected model
HistGradientBoosting ─┘
                              ↓
                    cost-aware threshold
                              ↓
                     untouched test set
                              ↓
                Streamlit risk-scoring UI
```

## Methodology

- Removes exact duplicate transactions before splitting to reduce cross-split leakage risk.
- Uses a **chronological split**, rather than randomly mixing later transactions into earlier training data.
- Compares a class-weighted logistic baseline, balanced Random Forest, and weighted HistGradientBoosting.
- Selects the model using **validation PR-AUC**, not test performance.
- Selects a review threshold on validation data with false negatives assigned a higher cost than false positives.
- Evaluates the selected model once on the held-out test period.
- Keeps the raw ~150 MB dataset and trained binary artifacts out of Git.

## Dataset

The project uses the **Credit Card Fraud Detection** benchmark published by the Machine Learning Group at Université Libre de Bruxelles / Worldline and distributed through Kaggle. The data covers European cardholders over two days in September 2013. `V1`–`V28` are PCA-transformed anonymized features; `Time` and `Amount` are the only non-PCA predictors.

Download `creditcard.csv` from Kaggle and place it at:

```text
data/raw/creditcard.csv
```

Raw data is intentionally ignored by Git.

## Quick start

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -e ".[dev]"
python scripts/bootstrap.py
streamlit run app.py
```

The bootstrap command validates/cleans the dataset, builds the SQLite database, trains and compares models, tunes the threshold, and saves evaluation artifacts.

## SQL analytics

`sql/analytics.sql` includes portfolio-ready queries for class imbalance, transaction amounts by class, and fraud activity by hour. The SQLite database is generated locally at `data/processed/fraudlens.db`.

## Tests and quality

```bash
python -m ruff check .
python -m pytest -q
```

GitHub Actions runs both checks on every push and pull request.

## Repository structure

```text
app.py                  Streamlit risk-scoring application
src/fraudlens/          reusable data, feature, model and evaluation code
scripts/                reproducible pipeline and CLI scoring
sql/                    analytics queries
tests/                  automated tests
docs/model_card.md      intended use, limitations and evaluation notes
data/                    ignored raw/processed data locations
models/                  ignored trained model artifacts
reports/                 generated evaluation outputs
```

## What this project demonstrates

Python · pandas · scikit-learn · imbalanced classification · PR-AUC · chronological validation · cost-sensitive thresholding · SQL/SQLite · Streamlit · testing · linting · CI · Docker

## Limitations

This benchmark is old, covers only ~48 hours, and hides the semantic meaning of V1–V28. Fraud patterns drift over time, so the model is **not** suitable for current real-world payment decisions. The application is a portfolio/research demonstration and should not be used to automatically block customers or transactions.

## Attribution

Dataset: Machine Learning Group — ULB / Worldline, *Credit Card Fraud Detection*, distributed via Kaggle. See the dataset page for its database/content licensing and source references.

## License

Project source code: MIT. Dataset licensing is separate and governed by the dataset publisher/distributor.
