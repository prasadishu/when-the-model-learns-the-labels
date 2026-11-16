# When the Model Learns the Label — analysis code

Code that regenerates **every number, table and figure** in the manuscript *"When the Model Learns the Label: A Machine
Learning Audit of Outcome-Circularity in Chronic Kidney and Cardiovascular Disease Risk Stratification for NAFLD"*
and its Supplementary Material (TRIPOD+AI item 18f).

## What is here

| Path | Purpose |
|---|---|
| `src/config.py` | Every constant (seed, folds, bootstrap size, panels, hyper-parameters) |
| `src/data.py` | Loading, feature panels, fold-safe preprocessing |
| `src/cvd_staging.py` | **Numeric CVD staging algorithm** (points + cut-offs) reconstructed from the verbal criteria |
| `src/models.py` | Six classical models + cross-validation driver |
| `src/tabnet_dt.py` | TabNet-family models (PyTorch): DT-XTabNet with Attention Divergence Regularisation, independent TabNets, Shared-Trunk TabNet, Shared-Bottom MLP |
| `src/stats_utils.py` | AUC, BCa bootstrap, DeLong test, paired bootstrap, NRI/IDI, calibration, decision curves |
| `scripts/01_staging_and_cohort.py` | Cohort checks, CKM identity, CVD staging reconstruction and its agreement with the recorded labels |
| `scripts/02_benchmark.py` | Six classical models x feature sets x outcomes (Table 1, Supplementary Table S2) |
| `scripts/03_audit_and_secondary.py` | Overlap audit (Table 2, S3), DeLong, reclassification (S4), subgroups (S5), calibration (S6), decision curves, correlations |
| `scripts/04_deep_learning.py` | TabNet-family benchmark, lambda sweep (S7), attention-vs-SHAP check |
| `scripts/05_make_figures.py` | Figures 1-4 and Supplementary Figures S1-S2 |
| `results/` | All generated tables (CSV/JSON) used in the manuscript |
| `figures/` | Generated figures |

## Run

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp /path/to/NAFLD_CKD_CVD_Indian_v2_all_NAFLD_Yes.xlsx data/      # patient-level data are NOT in this repository
bash run_all.sh                                                    # ~25 min on one CPU core
```

Everything is seeded (`SEED = 42`), uses a single CPU thread and 5-fold stratified cross-validation, so reruns reproduce
the reported figures exactly on the same library versions (see `requirements.txt`).

## Data

The analysis file is a 1,500-patient single-centre cohort (sheet `NAFLD_Dataset`, variable names in row 3, records from
row 4). It is **not** redistributed here; access arrangements are described in the manuscript's Data availability statement.
`NAFLD_US` is "Yes" for every record (all patients have ultrasound-defined NAFLD) and is therefore dropped automatically as a
constant column, leaving 46 predictors (35 continuous, 11 categorical).

## CVD staging algorithm (summary)

`CVD_Stage` was assigned from verbal criteria; no numeric algorithm existed. `src/cvd_staging.py` expresses them as a points
score (diabetes 3, hypertension 2, dyslipidaemia 1, metabolic syndrome 1, organ damage 2, +1 per other risk factor) with
cut-offs 6 / 8 / 9 / 11 for stages 1 / 2 / 3 / 4. It reproduces the recorded stage exactly in 44.4% of patients (81.7% within
one stage; 75.5% for the Stage >= 3 outcome used in the models). It is a reconstruction: weights are judgement-based and the
cut-offs were fitted to the recorded labels (they were identical in every cross-validation fold).

## Licence

MIT (see `LICENSE`). If you reuse the code, please cite the manuscript.
