"""Circularity audit (Table 2 / S3), DeLong tests, reclassification (S4), subgroups (S5), calibration (S6),
decision curves, cross-domain correlation and feature importances. Uses out-of-fold predictions from 02."""
import sys, os, json; sys.path.insert(0, os.getcwd())
import numpy as np, pandas as pd
from src import config as C, data as D, models as M, cvd_staging as S
from src.stats_utils import (auc, bca_ci, delong_test, paired_boot_delta, reclassification,
                             calibration, calibration_curve, net_benefit, treat_all)

df = D.load(); panels, _ = D.panels(df); fold_idx, _ = D.folds(df); ckd, cvd = D.outcomes(df)
Y = {"CKD": ckd, "CVD": cvd}; oof = dict(np.load(f"{C.RESULTS_DIR}/oof_predictions.npz"))
out = {}; GB = "Gradient Boosting"

# ---------- Table 2: overlap vs non-overlap under three partitions ----------
parts = {
    "R (reconstructed score; primary)": [v for v in C.CURATED if v in S.INPUTS],
    "P1 (earlier working partition)": C.P1_OVERLAP,
    "P2 (earlier working partition)": C.P2_OVERLAP,
}
t2, t3, sub_oof = [], [], {}
full14 = oof[f"curated|{GB}|CVD"]
a, lo, hi = bca_ci(cvd, full14); t2.append(dict(subset="Full curated panel (14 variables)", n_vars=14, auc=a, lo=lo, hi=hi))
for name, ov in parts.items():
    non = [v for v in C.CURATED if v not in ov]
    p_ov, _, _ = M.cv_oof(GB, df, ov, cvd, fold_idx); p_non, _, _ = M.cv_oof(GB, df, non, cvd, fold_idx)
    for lab, cols, p in [("overlapping", ov, p_ov), ("non-overlapping", non, p_non)]:
        a, lo, hi = bca_ci(cvd, p)
        t2.append(dict(subset=f"{name}: {len(cols)} {lab} variables ({', '.join(cols)})", n_vars=len(cols), auc=a, lo=lo, hi=hi))
    d1 = paired_boot_delta(cvd, p_ov, full14); d2 = paired_boot_delta(cvd, p_ov, p_non)
    t3 += [dict(comparison=f"{name}: overlap vs full panel", delta=d1[0], lo=d1[1], hi=d1[2]),
           dict(comparison=f"{name}: overlap vs non-overlap", delta=d2[0], lo=d2[1], hi=d2[2])]
pd.DataFrame(t2).to_csv(f"{C.RESULTS_DIR}/table2_overlap_audit.csv", index=False)
pd.DataFrame(t3).to_csv(f"{C.RESULTS_DIR}/tableS3_paired_bootstrap.csv", index=False)

# ---------- DeLong, 8 pre-specified comparisons, Bonferroni alpha = 0.05/8 ----------
cmp = [("curated|GB", "agesex|LR"), ("full|GB", "curated|GB"), ("full|GB", "nonrenal|GB"), ("nonrenal|GB", "curated|GB")]
key = lambda k, o: f"{k.split('|')[1] and ''}"
mp = {"GB": GB, "LR": "Logistic Regression"}
rows = []
for outcome, y in Y.items():
    for a_, b_ in cmp:
        pa = oof[f"{a_.split('|')[0]}|{mp[a_.split('|')[1]]}|{outcome}"]; pb = oof[f"{b_.split('|')[0]}|{mp[b_.split('|')[1]]}|{outcome}"]
        A, B, z, p = delong_test(y, pa, pb)
        rows.append(dict(outcome=outcome, a=a_, b=b_, auc_a=A, auc_b=B, z=z, p=p, significant_bonferroni=p < 0.05 / 8))
pd.DataFrame(rows).to_csv(f"{C.RESULTS_DIR}/delong_comparisons.csv", index=False)

# ---------- Reclassification (curated GB vs age+sex LR) ----------
rec = []
for outcome, y in Y.items():
    r = reclassification(y, oof[f"agesex|Logistic Regression|{outcome}"], oof[f"curated|{GB}|{outcome}"])
    for k, (e, l, h) in r.items(): rec.append(dict(outcome=outcome, metric=k, est=e, lo=l, hi=h))
pd.DataFrame(rec).to_csv(f"{C.RESULTS_DIR}/tableS4_reclassification.csv", index=False)

# ---------- Subgroups (curated GB) ----------
subs = {"Male": df.Sex == "Male", "Female": df.Sex == "Female",
        "Diabetic": df.Diabetes_Hx == "Yes", "Non-diabetic": df.Diabetes_Hx == "No"}
srows = []
for name, m in subs.items():
    m = m.values; r = dict(subgroup=name, n=int(m.sum()))
    for outcome, y in Y.items():
        a, lo, hi = bca_ci(y[m], oof[f"curated|{GB}|{outcome}"][m]); r[f"{outcome}_auc"], r[f"{outcome}_lo"], r[f"{outcome}_hi"] = a, lo, hi
    srows.append(r)
pd.DataFrame(srows).to_csv(f"{C.RESULTS_DIR}/tableS5_subgroups.csv", index=False)

# ---------- Calibration (full-panel GB; curated also reported) ----------
cal = []
for panel in ["full", "curated"]:
    for outcome, y in Y.items():
        cal.append(dict(panel=panel, outcome=outcome, **calibration(y, oof[f"{panel}|{GB}|{outcome}"])))
pd.DataFrame(cal).to_csv(f"{C.RESULTS_DIR}/tableS6_calibration.csv", index=False)
curves = {f"{o}": calibration_curve(y, oof[f"full|{GB}|{o}"]) for o, y in Y.items()}

# ---------- Decision curves ----------
dca = {}
for outcome, y in Y.items():
    dca[outcome] = dict(curated=net_benefit(y, oof[f"curated|{GB}|{outcome}"]),
                        agesex=net_benefit(y, oof[f"agesex|Logistic Regression|{outcome}"]), all=treat_all(y))
np.savez(f"{C.RESULTS_DIR}/figure_arrays.npz",
         **{f"cal_{k}": v for k, v in curves.items()},
         **{f"dca_{o}_{k}": v for o, d in dca.items() for k, v in d.items()})

# ---------- Cross-domain correlation among curated variables ----------
dom = {"Age": "demographic", "Sex": "demographic", "HbA1c_pct": "glycaemic", "HOMA_IR": "glycaemic",
       "ALT_UL": "hepatic", "AST_UL": "hepatic", "GGT_UL": "hepatic", "Triglycerides_mgdL": "lipid",
       "HDL_mgdL": "lipid", "LDL_mgdL": "lipid", "CRP_mgL": "inflammatory", "BMI": "anthropometric",
       "Waist_Circ_cm": "anthropometric", "SBP_mmHg": "blood pressure"}
X = df[C.CURATED].copy(); X["Sex"] = (X["Sex"] == "Male").astype(int)
cm = X.corr(method="pearson"); pairs = [(a, b) for i, a in enumerate(C.CURATED) for b in C.CURATED[i + 1:]]
cross = [abs(cm.loc[a, b]) for a, b in pairs if dom[a] != dom[b]]
out["cross_domain_pairs"] = len(cross); out["cross_domain_mean_abs_r"] = float(np.mean(cross)); out["cross_domain_max_abs_r"] = float(np.max(cross))
out["total_pairs"] = len(pairs)

# ---------- CKM group baseline table & ethnicity chi-square ----------
from scipy import stats
hi = ((df.CKD_Stage >= 3) | (df.CVD_Stage >= 3)).values
ct = pd.crosstab(df.Ethnicity, hi)
chi2, p, dof, _ = stats.chi2_contingency(ct, correction=False)
prev = (ct[True] / ct.sum(axis=1)).round(3)
out["ethnicity"] = dict(groups=int(ct.shape[0]), chi2=float(chi2), df=int(dof), p=float(p), prevalence_min=float(prev.min()), prevalence_max=float(prev.max()))
rows = []
for lab, col, kind in [("Age, years", "Age", "c"), ("Male sex, %", "Sex", "b:Male"), ("BMI, kg/m2", "BMI", "c"), ("SBP, mmHg", "SBP_mmHg", "c"),
                       ("HbA1c, %", "HbA1c_pct", "c"), ("LDL-C, mg/dL", "LDL_mgdL", "c"), ("HDL-C, mg/dL", "HDL_mgdL", "c"),
                       ("Triglycerides, mg/dL", "Triglycerides_mgdL", "c"), ("CRP, mg/L", "CRP_mgL", "c"), ("eGFR, mL/min/1.73m2", "eGFR_mLmin173m2", "c"),
                       ("Diabetes history, %", "Diabetes_Hx", "b:Yes"), ("Hypertension history, %", "Hypertension_Hx", "b:Yes")]:
    if kind == "c":
        a, b = df.loc[~hi, col], df.loc[hi, col]
        rows.append(dict(characteristic=lab, low=f"{a.mean():.1f}±{a.std():.1f}", high=f"{b.mean():.1f}±{b.std():.1f}", p=stats.ttest_ind(a, b, equal_var=False).pvalue))
    else:
        lv = kind.split(":")[1]; a, b = (df.loc[~hi, col] == lv), (df.loc[hi, col] == lv)
        tab = [[a.sum(), (~a).sum()], [b.sum(), (~b).sum()]]
        rows.append(dict(characteristic=lab, low=f"{100*a.mean():.1f}", high=f"{100*b.mean():.1f}", p=stats.chi2_contingency(tab, correction=False)[1]))
pd.DataFrame(rows).to_csv(f"{C.RESULTS_DIR}/tableS1_baseline_by_ckm.csv", index=False)
out["n_low"], out["n_high"] = int((~hi).sum()), int(hi.sum())

# ---------- Feature importances for Fig 3 (curated GB, fit on all data for display only) ----------
imp = {}
for outcome, y in Y.items():
    pp = D.Preprocessor().fit(df, C.CURATED); m = M.CLASSICAL[GB](); m.fit(pp.onehot(df), y)
    imp[outcome] = dict(zip(M.feature_names(pp), m.feature_importances_))
json.dump(imp, open(f"{C.RESULTS_DIR}/figure3_importances.json", "w"), indent=1)
json.dump(out, open(f"{C.RESULTS_DIR}/03_misc.json", "w"), indent=1)
# add logistic-regression row for the curated panel to Table 1 (sensitivity to model choice)
t1 = pd.read_csv(f"{C.RESULTS_DIR}/table1_discrimination.csv")
r = dict(feature_set="Curated panel (logistic regression)", n_vars=14, model="Logistic Regression")
for outcome, y in Y.items():
    a, lo, hi = bca_ci(y, oof[f"curated|Logistic Regression|{outcome}"]); r[f"{outcome}_auc"], r[f"{outcome}_lo"], r[f"{outcome}_hi"] = a, lo, hi
if not (t1.feature_set == r["feature_set"]).any():
    t1 = pd.concat([t1, pd.DataFrame([r])], ignore_index=True); t1.to_csv(f"{C.RESULTS_DIR}/table1_discrimination.csv", index=False)
print("done")
