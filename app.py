from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
import joblib
import pandas as pd
import streamlit as st

from fraudlens.config import METADATA_PATH, MODEL_PATH
from fraudlens.features import FEATURES, engineer_features

st.set_page_config(page_title="FraudLens", page_icon="🔎", layout="wide")
st.title("🔎 FraudLens")
st.caption("Explainable credit-card fraud risk scoring under extreme class imbalance")
st.warning(
    "Portfolio/research demo only. This system is not a production fraud decision engine and must not be used to automatically block real customers."
)
if not MODEL_PATH.exists():
    st.error(
        "Model artifact is missing. Place the ULB creditcard.csv in data/raw and run: python scripts/bootstrap.py"
    )
    st.stop()
artifact = joblib.load(MODEL_PATH)
model = artifact["model"]
threshold = float(artifact["threshold"])
meta = json.loads(METADATA_PATH.read_text()) if METADATA_PATH.exists() else {}
with st.sidebar:
    st.header("Model")
    st.write(meta.get("selected_model", "trained model").replace("_", " ").title())
    st.metric("Decision threshold", f"{threshold:.3f}")
    if meta.get("test"):
        st.metric("Test PR-AUC", f"{meta['test']['pr_auc']:.3f}")
        st.metric("Test recall", f"{meta['test']['recall']:.1%}")

tab1, tab2 = st.tabs(["Score a transaction", "Model & methodology"])
with tab1:
    st.subheader("Transaction features")
    st.caption(
        "V1–V28 are anonymized PCA components in the public ULB benchmark. Use a real dataset row for a meaningful demo."
    )
    cols = st.columns(4)
    row = {}
    for i in range(1, 29):
        row[f"V{i}"] = cols[(i - 1) % 4].number_input(f"V{i}", value=0.0, format="%.5f")
    a, b = st.columns(2)
    row["Amount"] = a.number_input("Amount", min_value=0.0, value=25.0)
    row["Time"] = b.number_input(
        "Time (seconds since first dataset transaction)", min_value=0.0, value=3600.0
    )
    if st.button("Analyze transaction", type="primary", use_container_width=True):
        x = engineer_features(pd.DataFrame([row]))[FEATURES]
        score = float(model.predict_proba(x)[0, 1])
        flagged = score >= threshold
        c1, c2, c3 = st.columns(3)
        c1.metric("Fraud risk", f"{score:.2%}")
        c2.metric("Threshold", f"{threshold:.2%}")
        c3.metric("Review decision", "FLAG" if flagged else "PASS")
        (st.error if flagged else st.success)(
            "Transaction exceeds the review threshold."
            if flagged
            else "Transaction is below the review threshold."
        )
        st.progress(min(max(score, 0.0), 1.0))
        if hasattr(model, "feature_importances_"):
            imp = (
                pd.DataFrame({"Feature": FEATURES, "Importance": model.feature_importances_})
                .sort_values("Importance", ascending=False)
                .head(12)
            )
            st.markdown("#### Global model importance")
            st.bar_chart(imp.set_index("Feature"))
with tab2:
    st.markdown("""### Why this project is different
Fraud is a rare-event problem. The benchmark contains only a tiny fraction of fraudulent transactions, so raw accuracy is not a useful headline metric. FraudLens prioritizes **precision, recall, F1 and PR-AUC**, selects models on validation data, and tunes the decision threshold separately from model training.

### Evaluation discipline
- exact duplicates are removed before splitting;
- transactions are split chronologically into train / validation / test partitions;
- the test partition is never used to select a model or threshold;
- threshold selection uses an explicit false-negative cost higher than the false-positive cost;
- raw cardholder fields are unavailable because the public benchmark exposes PCA components V1–V28.

### Responsible use
This is an educational portfolio system. Real fraud operations require drift monitoring, calibrated costs, investigator workflows, governance, privacy controls and human review.""")
