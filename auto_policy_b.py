
# auto_policy_b.py
# Auto-tune thresholds per Policy B (allow FN<=1 for optimal/efficiency; FN=0 for safe/conservative)
# Usage (run from your project root where src/metrics.py lives):
#   python auto_policy_b.py --csv data/your_predictions.csv --outdir cases
#
# Abbrev expanded once: FN = False Negative, FNR = False Negative Rate, WLR = Workload Reduction.
import argparse, sys, json
from pathlib import Path
import numpy as np
import pandas as pd

# Ensure project root (with src/) is on path when run from anywhere
cwd = Path.cwd()
if (cwd / "src").exists():
    sys.path.insert(0, str(cwd))
else:
    # also allow running from this script's directory if dropped into project root later
    sys.path.insert(0, str(Path(__file__).resolve().parent))

try:
    from src.metrics import compute_metrics  # type: ignore
except Exception as e:
    raise SystemExit("❌ Could not import src.metrics. Run this from your project root (where src/metrics.py exists).") from e

CASES = [
    {"name":"caseL_safe",         "assumed_prevalence":0.005, "target_fn":0},
    {"name":"caseL_optimal",      "assumed_prevalence":0.005, "target_fn":1},
    {"name":"caseM_safe",         "assumed_prevalence":0.015, "target_fn":0},
    {"name":"caseM_optimal",      "assumed_prevalence":0.015, "target_fn":1},
    {"name":"caseH_safe",         "assumed_prevalence":0.040, "target_fn":0},
    {"name":"caseH_optimal",      "assumed_prevalence":0.040, "target_fn":1},
    {"name":"caseE_efficiency",   "assumed_prevalence":0.007, "target_fn":1},
    {"name":"caseQ_conservative", "assumed_prevalence":0.005, "target_fn":0},
]

def pick_threshold_policy_b(y_true, y_score, target_fn):
    # Sweep thresholds and pick the HIGHEST threshold with FN <= target_fn (maximize WLR under the constraint).
    thrs = np.linspace(0.0, 1.0, 5001)  # step=0.0002
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score).astype(float)
    pos = (y_true == 1)

    best_thr = None
    for t in thrs:
        fn = int(((pos) & (y_score < t)).sum())
        if fn <= target_fn:
            best_thr = t  # keep going to get the highest t that still satisfies
    if best_thr is None:
        best_thr = 0.0
    return float(best_thr)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="data/your_predictions.csv",
                    help="CSV with columns: study_id, y_true (0/1), y_score (prob in [0,1])")
    ap.add_argument("--outdir", default="cases",
                    help="Output directory for case JSONs")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    assert {"y_true","y_score"}.issubset(df.columns), "CSV must have y_true and y_score columns"

    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)

    rows = []
    for case in CASES:
        thr = pick_threshold_policy_b(df["y_true"].values, df["y_score"].values, case["target_fn"])
        res = compute_metrics(df["y_true"], df["y_score"], thr, prevalence_assumed=case["assumed_prevalence"])
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
        outpath = outdir / f"{case['name']}.json"
        outpath.write_text(json.dumps(card, indent=2))
        rows.append((case["name"], thr, res.fn, res.fnr, res.wlr, res.npv))

    print("\nPolicy B tuning complete. Summary (name, threshold, FN, FNR, WLR, NPV):")
    for name, thr, fn, fnr, wlr, npv in rows:
        print(f"{name:22s}  thr={thr:.4f}  FN={fn:2d}  FNR={fnr*100:6.2f}%  WLR={wlr*100:6.2f}%  NPV={npv*100:6.2f}%")

if __name__ == "__main__":
    main()
