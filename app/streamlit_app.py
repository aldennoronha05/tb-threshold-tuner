import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import json
from pathlib import Path
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from src.metrics import compute_metrics

st.set_page_config(page_title="TB Threshold Tuner", layout="wide")
st.title("🫁 TB Threshold Tuner — Prevalence-Aware Triage")

default_csv = Path(__file__).resolve().parents[1] / "data" / "sample_predictions.csv"
uploaded = st.file_uploader("Upload predictions CSV (columns: study_id, y_true, y_score)", type=["csv"])
csv_path = uploaded if uploaded is not None else default_csv

df = pd.read_csv(csv_path)
if not set(['y_true','y_score']).issubset(df.columns):
    st.error("CSV must contain 'y_true' and 'y_score'."); st.stop()

st.sidebar.header("Controls")
thr = st.sidebar.slider("Decision threshold (positive if score ≥ threshold)", 0.0, 1.0, 0.5, 0.001)
prev_assumed = st.sidebar.slider("Assumed TB prevalence (%) for interpretation", 0.01, 5.0, 0.5, 0.01)

res = compute_metrics(df['y_true'], df['y_score'], thr, prevalence_assumed=prev_assumed/100.0)

left, right = st.columns(2)
with left:
    st.subheader("Confusion & Core Metrics")
    st.write({
        "TP": res.tp, "FP": res.fp, "TN": res.tn, "FN": res.fn,
        "Sensitivity": round(res.sensitivity,4),
        "Specificity": round(res.specificity,4),
        "FNR": round(res.fnr,6),
        "WLR": round(res.wlr,4),
        "PPV": round(res.ppv,4),
        "NPV": round(res.npv,6),
        "AUC": round(res.auc,4) if not np.isnan(res.auc) else None,
        "AUPRC": round(res.auprc,4) if not np.isnan(res.auprc) else None,
        "Net Benefit": round(res.net_benefit,6),
    })

with right:
    st.subheader("Threshold Sweep")
    thrs = np.linspace(0,1,501)
    fnrs = []; wlrs = []; sens = []; spec = []
    for t in thrs:
        r = compute_metrics(df['y_true'], df['y_score'], t)
        fnrs.append(r.fnr); wlrs.append(r.wlr); sens.append(r.sensitivity); spec.append(r.specificity)
    fig = plt.figure()
    plt.plot(thrs, fnrs, label="FNR")
    plt.plot(thrs, wlrs, label="WLR")
    plt.plot(thrs, sens, label="Sensitivity")
    plt.plot(thrs, spec, label="Specificity")
    plt.axvline(thr, linestyle="--")
    plt.xlabel("Threshold"); plt.ylabel("Value")
    plt.title("Trade-offs across thresholds")
    plt.legend(loc="best")
    st.pyplot(fig)

st.subheader("Export Threshold Card")
if st.button("Save JSON"):
    card = {
        "threshold": res.threshold,
        "assumed_prevalence": res.prevalence_assumed,
        "metrics": {
            "TP": res.tp, "FP": res.fp, "TN": res.tn, "FN": res.fn,
            "sensitivity": res.sensitivity, "specificity": res.specificity,
            "fnr": res.fnr, "wlr": res.wlr, "ppv": res.ppv, "npv": res.npv,
            "auc": res.auc, "auprc": res.auprc, "net_benefit": res.net_benefit
        }
    }
    out = Path("threshold_card.json")
    out.write_text(json.dumps(card, indent=2))
    st.success(f"Saved {out.resolve()}")
