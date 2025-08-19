from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score

@dataclass
class MetricsResult:
    threshold: float
    prevalence_assumed: float
    tn: int; fp: int; fn: int; tp: int
    sensitivity: float; specificity: float
    fnr: float; wlr: float
    ppv: float; npv: float
    auc: float; auprc: float
    net_benefit: float

def confusion_at_threshold(y_true, y_score, thr):
    y_pred = (y_score >= thr).astype(int)
    tp = int(((y_true==1)&(y_pred==1)).sum())
    fp = int(((y_true==0)&(y_pred==1)).sum())
    tn = int(((y_true==0)&(y_pred==0)).sum())
    fn = int(((y_true==1)&(y_pred==0)).sum())
    return tn, fp, fn, tp

def safe_div(a, b):
    return a / b if b else 0.0

def compute_metrics(y_true, y_score, thr: float, prevalence_assumed: float= None):
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score).astype(float)
    tn, fp, fn, tp = confusion_at_threshold(y_true, y_score, thr)
    n = len(y_true)
    sens = safe_div(tp, tp+fn)
    spec = safe_div(tn, tn+fp)
    fnr  = safe_div(fn, tp+fn)
    wlr  = safe_div(tn+fn, n)
    ppv  = safe_div(tp, tp+fp)
    npv  = safe_div(tn, tn+fn)
    try:
        auc = roc_auc_score(y_true, y_score)
        auprc = average_precision_score(y_true, y_score)
    except Exception:
        auc = float('nan'); auprc = float('nan')
    t = thr if 0 < thr < 1 else 0.5
    net_benefit = (tp/n) - (fp/n)*(t/(1-t))
    return MetricsResult(threshold=thr, prevalence_assumed=(prevalence_assumed if prevalence_assumed is not None else float(np.mean(y_true))),
                         tn=tn, fp=fp, fn=fn, tp=tp,
                         sensitivity=sens, specificity=spec,
                         fnr=fnr, wlr=wlr, ppv=ppv, npv=npv,
                         auc=auc, auprc=auprc, net_benefit=net_benefit)
