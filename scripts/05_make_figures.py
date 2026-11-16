"""Figures 1-4 and Supplementary Figures S1-S2 (generated only from files in results/)."""
import sys, os, json; sys.path.insert(0, os.getcwd())
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
from src import config as C, data as D

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
os.makedirs(C.FIG_DIR, exist_ok=True); R = C.RESULTS_DIR
df = D.load(); ckd, cvd = D.outcomes(df); Y = {"CKD": ckd, "CVD": cvd}
oof = dict(np.load(f"{R}/oof_predictions.npz")); arr = np.load(f"{R}/figure_arrays.npz")
GB, LR = "Gradient Boosting", "Logistic Regression"
t1 = pd.read_csv(f"{R}/table1_discrimination.csv")

# Figure 1
fig, ax = plt.subplots(figsize=(7.5, 3.6)); sets = t1[t1.feature_set != "Curated panel (logistic regression)"]; x = np.arange(len(sets)); w = 0.38
for j, (o, col) in enumerate([("CKD", "#1f4e79"), ("CVD", "#c55a11")]):
    a = sets[f"{o}_auc"].values; ax.bar(x + (j - .5) * w, a, w, label=o, color=col,
        yerr=[a - sets[f"{o}_lo"].values, sets[f"{o}_hi"].values - a], capsize=2, ecolor="k")
ax.set_xticks(x); ax.set_xticklabels([f"{s}\n({n} var.)" for s, n in zip(sets.feature_set, sets.n_vars)], fontsize=7)
ax.axhline(0.5, color="grey", ls=":", lw=.8); ax.set_ylabel("AUC (pooled out-of-fold, 95% BCa CI)"); ax.set_ylim(0.4, 1.02); ax.legend(frameon=False)
fig.tight_layout(); fig.savefig(f"{C.FIG_DIR}/figure1_discrimination.png", dpi=300); plt.close(fig)

# Figure 2
fig, axs = plt.subplots(1, 2, figsize=(7.5, 3.6))
for ax, o in zip(axs, ["CKD", "CVD"]):
    for lab, key, ls in [("Full panel", f"full|{GB}|{o}", "-"), ("Curated 14-variable panel", f"curated|{GB}|{o}", "--"), ("Age + sex", f"agesex|{LR}|{o}", ":")]:
        fpr, tpr, _ = roc_curve(Y[o], oof[key]); ax.plot(fpr, tpr, ls, label=lab)
    ax.plot([0, 1], [0, 1], color="grey", lw=.6); ax.set_title(o); ax.set_xlabel("1 - specificity"); ax.set_ylabel("Sensitivity")
axs[0].legend(frameon=False, fontsize=7, loc="lower right"); fig.tight_layout(); fig.savefig(f"{C.FIG_DIR}/figure2_roc.png", dpi=300); plt.close(fig)

# Figure 3
imp = json.load(open(f"{R}/figure3_importances.json")); fig, axs = plt.subplots(1, 2, figsize=(7.5, 3.6))
for ax, o in zip(axs, ["CKD", "CVD"]):
    s = pd.Series(imp[o]).sort_values().tail(10); ax.barh(s.index, s.values, color="#1f4e79" if o == "CKD" else "#c55a11"); ax.set_title(f"{o}: gradient-boosting importance"); ax.tick_params(labelsize=7)
fig.tight_layout(); fig.savefig(f"{C.FIG_DIR}/figure3_importance.png", dpi=300); plt.close(fig)

# Figure 4
fig, axs = plt.subplots(1, 2, figsize=(7.5, 3.6))
for ax, o in zip(axs, ["CKD", "CVD"]):
    c = arr[f"cal_{o}"]; ax.plot([0, 1], [0, 1], color="grey", lw=.6); ax.plot(c[:, 0], c[:, 1], "o-"); ax.set_title(f"{o} (full panel)"); ax.set_xlabel("Predicted probability"); ax.set_ylabel("Observed proportion")
fig.tight_layout(); fig.savefig(f"{C.FIG_DIR}/figure4_calibration.png", dpi=300); plt.close(fig)

# Figure S1
s5 = pd.read_csv(f"{R}/tableS5_subgroups.csv"); fig, axs = plt.subplots(1, 2, figsize=(7.5, 4.14), sharey=True)
for ax, o in zip(axs, ["CKD", "CVD"]):
    y = np.arange(len(s5))[::-1]; ax.errorbar(s5[f"{o}_auc"], y, xerr=[s5[f"{o}_auc"] - s5[f"{o}_lo"], s5[f"{o}_hi"] - s5[f"{o}_auc"]], fmt="o", color="k", capsize=2)
    ax.set_yticks(y); ax.set_yticklabels([f"{a} (n={n})" for a, n in zip(s5.subgroup, s5.n)]); ax.set_title(o); ax.set_xlabel("AUC (95% CI)"); ax.axvline(.5, color="grey", ls=":")
fig.tight_layout(); fig.savefig(f"{C.FIG_DIR}/figureS1_subgroups.png", dpi=300); plt.close(fig)

# Figure S2
fig, axs = plt.subplots(1, 2, figsize=(7.5, 3.44)); th = C.DCA_THRESHOLDS
for ax, o in zip(axs, ["CKD", "CVD"]):
    ax.plot(th, arr[f"dca_{o}_curated"], label="Curated panel"); ax.plot(th, arr[f"dca_{o}_agesex"], "--", label="Age + sex"); ax.plot(th, arr[f"dca_{o}_all"], ":", label="Treat all"); ax.axhline(0, color="k", lw=.6)
    ax.set_title(o); ax.set_xlabel("Threshold probability"); ax.set_ylabel("Net benefit"); ax.set_ylim(-0.05, max(0.3, arr[f"dca_{o}_curated"].max() * 1.1))
axs[0].legend(frameon=False, fontsize=7); fig.tight_layout(); fig.savefig(f"{C.FIG_DIR}/figureS2_decision_curves.png", dpi=300); plt.close(fig)
print("figures written")
