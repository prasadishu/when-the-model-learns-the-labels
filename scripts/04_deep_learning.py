"""DT-XTabNet (+/- ADR), lambda sweep, independent TabNets, Shared-Trunk TabNet, Shared-Bottom MLP; attention-vs-SHAP check."""
import sys, os, json, time; sys.path.insert(0, os.getcwd())
import numpy as np, pandas as pd, torch
from scipy import stats
from src import config as C, data as D
from src.tabnet_dt import TabNetFamily, SharedBottomMLP, fit_predict, n_params
from src.stats_utils import auc, bca_ci

QUICK = os.environ.get("QUICK") == "1"            # smoke test: 1 fold, 5 epochs
cfg = dict(C.TABNET); cfg["max_epochs"] = 5 if QUICK else cfg["max_epochs"]
df = D.load(); panels, _ = D.panels(df); cols = panels["full"]
fold_idx, strat = D.folds(df); ckd, cvd = D.outcomes(df); Y = np.stack([ckd, cvd], 1).astype(np.float32)
fold_idx = fold_idx[:1] if QUICK else fold_idx
res_rows, oofs, keep = [], {}, {}

def run(name, topology, lam=0.0, tasks=(0, 1), mlp=False):
    oof = np.full((len(df), 2), np.nan); fa, cos, eps, params = [], [], [], None
    for k, (tr, te) in enumerate(fold_idx):
        pp = D.Preprocessor().fit(df.iloc[tr], cols)
        ctr, ktr, cte, kte = pp.cont(df.iloc[tr]), pp.codes(df.iloc[tr]), pp.cont(df.iloc[te]), pp.codes(df.iloc[te])
        if mlp:
            mk = lambda: SharedBottomMLP(pp.onehot(df.iloc[tr]).shape[1])
            p, c, m, ep = fit_predict(mk, pp.onehot(df.iloc[tr]), ktr, Y[tr], pp.onehot(df.iloc[te]), kte, lam, C.SEED + k,
                                      lr=1e-3, cfg=cfg, strat=strat[tr])
            oof[te] = p
        elif topology == "single":
            ps, ms, eps_k = [], [], []
            for t in tasks:
                mk = lambda: TabNetFamily(ctr.shape[1], pp.cat_cardinalities, "single", **{x: cfg[x] for x in ["n_d", "n_a", "gamma", "emb_dim", "dropout", "vbs"]})
                p, c, m, ep = fit_predict(mk, ctr, ktr, Y[tr][:, [t]], cte, kte, 0.0, C.SEED + k, cfg=cfg, strat=strat[tr])
                oof[te, t] = p[:, 0]; ms.append(m); eps_k.append(ep)
            m, ep, c = ms, int(np.mean(eps_k)), float("nan")
        else:
            mk = lambda: TabNetFamily(ctr.shape[1], pp.cat_cardinalities, topology, **{x: cfg[x] for x in ["n_d", "n_a", "gamma", "emb_dim", "dropout", "vbs"]},
                                      n_shared_steps=cfg["n_shared_steps"], n_branch_steps=cfg["n_branch_steps"])
            p, c, m, ep = fit_predict(mk, ctr, ktr, Y[tr], cte, kte, lam, C.SEED + k, cfg=cfg, strat=strat[tr])
            oof[te] = p
        if k == 0:
            keep[name] = (m, pp, tr, te); params = sum(n_params(x) for x in m) if isinstance(m, list) else n_params(m)
        fa.append([auc(Y[te][:, t], oof[te, t]) for t in (0, 1)]); cos.append(c); eps.append(ep)
    fa = np.array(fa)
    r = dict(model=name, lam=lam, params=params, ckd_auc=fa[:, 0].mean(), ckd_sd=fa[:, 0].std(ddof=1) if len(fa) > 1 else np.nan,
             cvd_auc=fa[:, 1].mean(), cvd_sd=fa[:, 1].std(ddof=1) if len(fa) > 1 else np.nan,
             attention_cosine=float(np.nanmean(cos)), mean_epochs=float(np.mean(eps)))
    res_rows.append(r); oofs[name] = oof
    print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()}, flush=True)

t0 = time.time()
for lam in C.LAMBDA_GRID:
    run(f"DT-XTabNet lambda={lam:.2f}", "dt", lam)
sweep = pd.DataFrame(res_rows)
# lambda-selection rule (fixed a priori): smallest lambda with >=90% cosine reduction vs lambda=0 and AUC drop <=0.005 on both tasks
base = sweep.iloc[0]; ok = sweep[(sweep.attention_cosine <= 0.10 * base.attention_cosine) & (sweep.ckd_auc >= base.ckd_auc - 0.005) & (sweep.cvd_auc >= base.cvd_auc - 0.005)]
selected_by_rule = bool(len(ok))
sel = float(ok.iloc[0].lam) if len(ok) else float(sweep.iloc[sweep.attention_cosine.idxmin()].lam)   # fallback: largest cosine reduction
print("selected lambda:", sel, "(met both criteria)" if selected_by_rule else "(no lambda met both criteria; fallback = largest cosine reduction)", flush=True)
run("Two independent TabNets", "single"); run("Shared-Trunk TabNet", "shared"); run("Shared-Bottom MLP", "mlp", mlp=True)
tab = pd.DataFrame(res_rows); tab["selected_lambda"] = sel; tab["selected_by_rule"] = selected_by_rule
tab.to_csv(f"{C.RESULTS_DIR}/04_deep_learning{'_quick' if QUICK else ''}.csv", index=False)
np.savez_compressed(f"{C.RESULTS_DIR}/oof_deep{'_quick' if QUICK else ''}.npz", **{k.replace(' ', '_'): v for k, v in oofs.items()})

# ---------- attention vs SHAP (selected lambda, fold 1 test set) ----------
if not QUICK:
    def permutation_shap(fn, X, bg, n_perm=40, seed=C.SEED):
        """Interventional Shapley values by permutation sampling (marginal contributions along random feature orders)."""
        rng = np.random.default_rng(seed); n, Fd = X.shape; phi = np.zeros((n, Fd, 2))
        for i in range(n):
            for _ in range(n_perm):
                order = rng.permutation(Fd); rows = np.tile(bg[rng.integers(len(bg))], (Fd + 1, 1))
                for k, j in enumerate(order):
                    rows[k + 1:, j] = X[i, j]
                pr = fn(rows); phi[i, order] += (pr[1:] - pr[:-1])
        return phi / n_perm
    name = f"DT-XTabNet lambda={sel:.2f}"; model, pp, tr, te = keep[name]; model.eval()
    ctr, ktr = pp.cont(df.iloc[tr]), pp.codes(df.iloc[tr]); cte, kte = pp.cont(df.iloc[te]), pp.codes(df.iloc[te]); nc = ctr.shape[1]
    f = lambda x: torch.sigmoid(model(torch.as_tensor(x[:, :nc], dtype=torch.float32), torch.as_tensor(np.rint(x[:, nc:]), dtype=torch.long))[0]).detach().numpy()
    rng = np.random.default_rng(C.SEED); bg = np.hstack([ctr, ktr])[rng.choice(len(tr), 30, replace=False)]
    sel_i = rng.choice(len(te), 50, replace=False); Xs = np.hstack([cte, kte])[sel_i]
    with torch.no_grad():
        _, aggs = model(torch.as_tensor(cte), torch.as_tensor(kte, dtype=torch.long))
    sv = permutation_shap(f, Xs, bg)                                         # (n, F, 2)
    names = pp.num + pp.cat; out = {}
    for t, lab in enumerate(["CKD", "CVD"]):
        att = aggs[t].mean(0).numpy(); shp = np.abs(sv[:, :, t]).mean(0); keepf = att >= 0.01
        rho, p = stats.spearmanr(att[keepf], shp[keepf])
        out[lab] = dict(rho=float(rho), p=float(p), n_features=int(keepf.sum()), top_attention=[names[i] for i in np.argsort(-att)[:5]],
                        top_shap=[names[i] for i in np.argsort(-shp)[:5]])
    json.dump(out, open(f"{C.RESULTS_DIR}/04_attention_vs_shap.json", "w"), indent=1); print(out)
json.dump(dict(selected_lambda=sel, selected_by_rule=selected_by_rule, runtime_s=time.time() - t0), open(f"{C.RESULTS_DIR}/04_meta{'_quick' if QUICK else ''}.json", "w"))
