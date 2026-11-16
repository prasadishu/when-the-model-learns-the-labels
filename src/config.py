"""Central configuration for the analysis pipeline (single source of truth for every constant)."""
import os

SEED = 42
N_SPLITS = 5
N_BOOT = 2000
DATA_PATH = os.environ.get("NAFLD_DATA", "data/NAFLD_CKD_CVD_Indian_v2_all_NAFLD_Yes.xlsx")
RESULTS_DIR = os.environ.get("NAFLD_RESULTS", "results")
FIG_DIR = os.environ.get("NAFLD_FIGS", "figures")

ID_COL = "Patient_ID"
LABELS = ["CKD_Stage", "CVD_Stage", "CKM_Stage"]
HIGH_RISK_STAGE = 3                      # outcome = stage >= 3

# Ten continuous variables with |skewness| > 1 -> log1p (fixed a priori, identical in every fold)
LOG1P_VARS = ["ALT_UL", "AST_UL", "GGT_UL", "UACR_mgG", "CRP_mgL", "Insulin_uIUmL",
              "HOMA_IR", "Triglycerides_mgdL", "Serum_Creatinine_mgdL", "FIB4_Score"]

# Renal markers removed to build the non-renal panel
RENAL = ["Serum_Creatinine_mgdL", "eGFR_mLmin173m2", "UACR_mgG", "BUN_mgdL"]

# Curated 14-variable metabolic/hepatic panel
CURATED = ["HbA1c_pct", "HOMA_IR", "ALT_UL", "AST_UL", "GGT_UL", "Triglycerides_mgdL",
           "HDL_mgdL", "LDL_mgdL", "CRP_mgL", "BMI", "Waist_Circ_cm", "SBP_mmHg", "Age", "Sex"]
AGE_SEX = ["Age", "Sex"]

# Variable partitions used in the circularity audit.
# R  : derived programmatically from the inputs of the reconstructed CVD score (src/cvd_staging.py)
# P1/P2 : earlier working partitions (variable lists only; the formulae behind them are not available)
P1_OVERLAP = ["Age", "SBP_mmHg", "LDL_mgdL", "HDL_mgdL", "Triglycerides_mgdL", "BMI", "HbA1c_pct"]
P2_OVERLAP = ["Age", "SBP_mmHg", "BMI", "HDL_mgdL", "HbA1c_pct", "CRP_mgL", "Sex"]

# Net-reclassification risk categories and decision-curve thresholds
NRI_CUTS = (0.10, 0.20)
DCA_THRESHOLDS = [round(0.05 + 0.01 * i, 2) for i in range(56)]   # 0.05 .. 0.60

# DT-XTabNet hyper-parameters (fixed a priori; no tuning)
TABNET = dict(n_d=32, n_a=32, gamma=1.5, n_shared_steps=2, n_branch_steps=3, emb_dim=4,
              dropout=0.1, vbs=32, batch=128, lr=0.01, wd=1e-5, patience=15, max_epochs=150,
              val_frac=0.15)
LAMBDA_GRID = [0.0, 0.10, 0.25, 0.40, 0.50]
