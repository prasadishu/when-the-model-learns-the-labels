"""Cohort checks, CKD/CKM label identities, and the reconstructed CVD staging algorithm (+ optimism-corrected agreement)."""
import sys, json, os; sys.path.insert(0, os.getcwd())
import numpy as np, pandas as pd
from sklearn.model_selection import StratifiedKFold
from src import config as C, data as D, cvd_staging as S
from pandas.api.types import is_numeric_dtype

df = D.load(); os.makedirs(C.RESULTS_DIR, exist_ok=True)
panels, dropped = D.panels(df)
ckd, cvd = D.outcomes(df)
res = {"n": len(df), "n_predictors_after_dropping_constants": len(panels["full"]),
       "dropped_constant_columns": dropped, "n_nonrenal": len(panels["nonrenal"]),
       "renal_markers_removed": C.RENAL, "n_continuous": int(sum(is_numeric_dtype(df[c]) for c in panels["full"])),
       "n_categorical": int(sum(not is_numeric_dtype(df[c]) for c in panels["full"]))}
res["prevalence_ckd_high"] = float(ckd.mean()); res["prevalence_cvd_high"] = float(cvd.mean())
ckm_high = ((df.CKD_Stage >= 3) | (df.CVD_Stage >= 3))
res["ckm_high_n"] = int(ckm_high.sum()); res["ckm_equals_max_all_rows"] = bool((df.CKM_Stage == df[["CKD_Stage", "CVD_Stage"]].max(axis=1)).all())
res["ckm_ge3_matches_high"] = bool(((df.CKM_Stage >= 3) == ckm_high).all())
sk = df[[c for c in panels["full"] if is_numeric_dtype(df[c])]].skew()
res["log1p_vars_abs_skew"] = {v: round(float(sk[v]), 2) for v in C.LOG1P_VARS}
res["other_vars_abs_skew_gt1"] = [c for c in sk.index if abs(sk[c]) > 1 and c not in C.LOG1P_VARS]

# CVD reconstruction
s = S.score(df); recorded = df.CVD_Stage.values
res["cvd_weights"] = S.WEIGHTS; res["cvd_cutoffs"] = list(S.CUTOFFS)
res["cvd_reconstruction_in_sample"] = S.agreement(S.stage_from_score(s), recorded)
res["cvd_reconstruction_refit_cutoffs"] = dict(zip(["cutoffs", "exact"], map(lambda v: v if not isinstance(v, tuple) else list(map(int, v)), S.fit_cutoffs(s, recorded))))
# optimism-corrected agreement: cut-offs fitted on 4 folds, scored on the held-out fold
skf = StratifiedKFold(C.N_SPLITS, shuffle=True, random_state=C.SEED); pred = np.zeros(len(df), int)
for tr, te in skf.split(s, recorded):
    cut, _ = S.fit_cutoffs(s[tr], recorded[tr]); pred[te] = S.stage_from_score(s[te], cut)
res["cvd_reconstruction_cv_out_of_fold"] = S.agreement(pred, recorded)
res["cvd_recorded_distribution"] = np.bincount(recorded, minlength=5).tolist()
res["cvd_reconstructed_distribution"] = np.bincount(S.stage_from_score(s), minlength=5).tolist()
res["cvd_confusion_recorded_by_reconstructed"] = pd.crosstab(recorded, S.stage_from_score(s)).values.tolist()
# stage >=3 (outcome used in the models) agreement
res["cvd_high_risk_agreement"] = float(np.mean((S.stage_from_score(s) >= 3) == (recorded >= 3)))
# overlap partition implied by the reconstructed score
res["R_overlap"] = [v for v in C.CURATED if v in S.INPUTS]; res["R_nonoverlap"] = [v for v in C.CURATED if v not in S.INPUTS]
json.dump(res, open(f"{C.RESULTS_DIR}/01_staging_and_cohort.json", "w"), indent=2)
S.components(df).assign(Patient_ID=df.Patient_ID, score=s, reconstructed_stage=S.stage_from_score(s), recorded_stage=recorded)\
    .to_csv(f"{C.RESULTS_DIR}/cvd_score_per_patient.csv", index=False)
print(json.dumps(res, indent=1))
