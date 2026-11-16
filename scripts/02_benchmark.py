"""Six classical models x feature sets x outcomes, 5-fold CV. Saves out-of-fold predictions for every later step."""
import sys, os, json, time; sys.path.insert(0, os.getcwd())
import numpy as np, pandas as pd
from src import config as C, data as D, models as M
from src.stats_utils import auc, bca_ci

df = D.load(); os.makedirs(C.RESULTS_DIR, exist_ok=True)
panels, _ = D.panels(df); panels["egfr_alone"] = ["eGFR_mLmin173m2"]
fold_idx, _ = D.folds(df); ckd, cvd = D.outcomes(df); Y = {"CKD": ckd, "CVD": cvd}

plan = [(m, p) for p in ["full", "nonrenal", "curated"] for m in M.CLASSICAL] + \
       [("Logistic Regression", "agesex"), ("Logistic Regression", "egfr_alone"),
        ("Gradient Boosting", "agesex")]
rows, oof_store = [], {}
for model, panel in plan:
    for outcome, y in Y.items():
        t = time.time()
        oof, fa, _ = M.cv_oof(model, df, panels[panel], y, fold_idx)
        rows.append(dict(model=model, panel=panel, n_vars=len(panels[panel]), outcome=outcome,
                         fold_mean_auc=fa.mean(), fold_sd_auc=fa.std(ddof=1), pooled_oof_auc=auc(y, oof)))
        oof_store[f"{panel}|{model}|{outcome}"] = oof
        print(f"{panel:10s} {model:20s} {outcome} fold {fa.mean():.3f}±{fa.std(ddof=1):.3f} pooled {auc(y, oof):.3f}  ({time.time()-t:.0f}s)", flush=True)
pd.DataFrame(rows).to_csv(f"{C.RESULTS_DIR}/02_benchmark_classical.csv", index=False)
np.savez_compressed(f"{C.RESULTS_DIR}/oof_predictions.npz", **oof_store)

# Table 1 (reference model: gradient boosting; low-dimensional baselines: logistic regression)
t1 = []
for label, panel, model in [("Full panel", "full", "Gradient Boosting"),
                            ("Broad non-renal panel", "nonrenal", "Gradient Boosting"),
                            ("eGFR alone", "egfr_alone", "Logistic Regression"),
                            ("Curated metabolic/hepatic panel", "curated", "Gradient Boosting"),
                            ("Age + sex only", "agesex", "Logistic Regression")]:
    r = dict(feature_set=label, n_vars=len(panels[panel]), model=model)
    for outcome, y in Y.items():
        a, lo, hi = bca_ci(y, oof_store[f"{panel}|{model}|{outcome}"])
        r[f"{outcome}_auc"], r[f"{outcome}_lo"], r[f"{outcome}_hi"] = a, lo, hi
    t1.append(r)
pd.DataFrame(t1).to_csv(f"{C.RESULTS_DIR}/table1_discrimination.csv", index=False)
print(pd.DataFrame(t1).round(3).to_string())
