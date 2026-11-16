"""Numeric reconstruction of the CVD staging criteria.

The study's CVD_Stage was assigned from verbal criteria (ACC/AHA risk-adapted); no numeric algorithm
existed. This module expresses those criteria as a points score with explicit cut-offs. It is a
*reconstruction*: the weights are judgement-based and the cut-offs are fitted to the recorded labels.

Verbal criteria                                   -> numeric component
 Stage 0  SBP<120, no DM, no risk factor
 Stage 1  1-2 moderate risk factors                -> 'other risk factors' count (1 point each)
 Stage 2  metabolic syndrome, >=3 risk factors     -> MetS (>=3 NCEP criteria) 1 point
 Stage 3  DM + HTN + dyslipidaemia                 -> DM 3, HTN 2, dyslipidaemia 1
 Stage 4  organ damage / clinical CVD              -> organ damage (eGFR<60 or UACR>=300) 2 points
"""
import itertools, numpy as np, pandas as pd

WEIGHTS = dict(dm=3, htn=2, dyslipidaemia=1, mets=1, organ_damage=2, other_rf=1)
CUTOFFS = (6, 8, 9, 11)          # stage 1 if score>=6, 2 if >=8, 3 if >=9, 4 if >=11

# every raw variable read by the score (used to derive the overlap partition programmatically)
INPUTS = ["Diabetes_Hx", "Fasting_Glucose_mgdL", "HbA1c_pct", "Hypertension_Hx", "SBP_mmHg", "DBP_mmHg",
          "LDL_mgdL", "Triglycerides_mgdL", "HDL_mgdL", "Statin_Use", "Sex", "Waist_Circ_cm",
          "eGFR_mLmin173m2", "UACR_mgG", "Age", "Smoking_Status", "BMI", "Non_HDL_mgdL",
          "Family_Hx_CVD", "Physical_Activity", "CRP_mgL"]


def components(df):
    male = df["Sex"] == "Male"
    low_hdl = np.where(male, df["HDL_mgdL"] < 40, df["HDL_mgdL"] < 50)
    dm = (df["Diabetes_Hx"] == "Yes") | (df["Fasting_Glucose_mgdL"] >= 126) | (df["HbA1c_pct"] >= 6.5)
    htn = (df["Hypertension_Hx"] == "Yes") | (df["SBP_mmHg"] >= 140) | (df["DBP_mmHg"] >= 90)
    dysl = (df["LDL_mgdL"] >= 130) | (df["Triglycerides_mgdL"] >= 150) | low_hdl | (df["Statin_Use"] == "Yes")
    mets_n = (np.where(male, df["Waist_Circ_cm"] >= 90, df["Waist_Circ_cm"] >= 80).astype(int)
              + (df["Triglycerides_mgdL"] >= 150).astype(int) + low_hdl.astype(int)
              + ((df["SBP_mmHg"] >= 130) | (df["DBP_mmHg"] >= 85) | (df["Hypertension_Hx"] == "Yes")).astype(int)
              + ((df["Fasting_Glucose_mgdL"] >= 100) | dm).astype(int))
    organ = (df["eGFR_mLmin173m2"] < 60) | (df["UACR_mgG"] >= 300)
    rf = ((df["Age"] >= 55).astype(int) + (df["Smoking_Status"] == "Current").astype(int)
          + (df["BMI"] >= 30).astype(int)
          + ((df["LDL_mgdL"] >= 130) | (df["Non_HDL_mgdL"] >= 160)).astype(int)
          + (df["Family_Hx_CVD"] == "Yes").astype(int)
          + (df["Physical_Activity"] == "Sedentary").astype(int) + (df["CRP_mgL"] >= 3).astype(int))
    return pd.DataFrame(dict(dm=dm.astype(int), htn=htn.astype(int), dyslipidaemia=dysl.astype(int),
                             mets=(mets_n >= 3).astype(int), organ_damage=organ.astype(int), other_rf=rf))


def score(df, weights=WEIGHTS):
    comp = components(df)
    return sum(weights[k] * comp[k] for k in weights).values


def stage_from_score(s, cutoffs=CUTOFFS):
    return np.digitize(s, cutoffs)


def agreement(pred, recorded):
    pred, recorded = np.asarray(pred), np.asarray(recorded)
    return dict(exact=float(np.mean(pred == recorded)), within_one=float(np.mean(np.abs(pred - recorded) <= 1)))


def fit_cutoffs(s, recorded):
    """Exhaustive search over ordered integer cut-offs maximising exact agreement."""
    levels = sorted(set(s))[1:]
    best, arg = -1, None
    for cut in itertools.combinations(levels, 4):
        m = np.mean(np.digitize(s, cut) == recorded)
        if m > best:
            best, arg = m, cut
    return arg, best
