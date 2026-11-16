"""Statistical utilities: AUC, BCa bootstrap CI, DeLong test, paired bootstrap, NRI/IDI, calibration, DCA."""
import numpy as np
from scipy import stats
import statsmodels.api as sm
from . import config as C


def auc(y, p):
    y = np.asarray(y); p = np.asarray(p)
    n1 = y.sum(); n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return np.nan
    r = stats.rankdata(p)
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def bca_ci(y, p, B=C.N_BOOT, seed=C.SEED, alpha=0.05):
    """BCa bootstrap CI for AUC (case resampling; jackknife acceleration)."""
    y = np.asarray(y); p = np.asarray(p); n = len(y)
    rng = np.random.default_rng(seed)
    theta = auc(y, p)
    boots = np.empty(B)
    for b in range(B):
        i = rng.integers(0, n, n)
        boots[b] = auc(y[i], p[i])
    boots = boots[~np.isnan(boots)]
    z0 = stats.norm.ppf(np.clip(np.mean(boots < theta), 1e-6, 1 - 1e-6))
    jack = np.array([auc(np.delete(y, i), np.delete(p, i)) for i in range(n)])
    d = jack.mean() - jack
    a = (d ** 3).sum() / (6.0 * ((d ** 2).sum() ** 1.5))
    zs = stats.norm.ppf([alpha / 2, 1 - alpha / 2])
    adj = stats.norm.cdf(z0 + (z0 + zs) / (1 - a * (z0 + zs)))
    lo, hi = np.quantile(boots, adj)
    return float(theta), float(lo), float(hi)


# ---------- DeLong (Sun & Xu 2014 fast implementation) ----------
def _midrank(x):
    j = np.argsort(x); z = x[j]; n = len(x); t = np.zeros(n); i = 0
    while i < n:
        k = i
        while k < n and z[k] == z[i]:
            k += 1
        t[i:k] = 0.5 * (i + k - 1) + 1
        i = k
    out = np.empty(n); out[j] = t
    return out


def _fast_delong(preds, m):
    k, n = preds.shape; nn = n - m
    pos, neg = preds[:, :m], preds[:, m:]
    tx = np.array([_midrank(pos[r]) for r in range(k)])
    ty = np.array([_midrank(neg[r]) for r in range(k)])
    tz = np.array([_midrank(preds[r]) for r in range(k)])
    aucs = tz[:, :m].sum(axis=1) / m / nn - (m + 1.0) / 2.0 / nn
    v01 = (tz[:, :m] - tx) / nn
    v10 = 1.0 - (tz[:, m:] - ty) / m
    cov = np.cov(v01) / m + np.cov(v10) / nn
    return aucs, cov


def delong_test(y, p1, p2):
    """Two-sided DeLong test for two correlated AUCs. Returns (auc1, auc2, z, p)."""
    y = np.asarray(y); order = np.argsort(-y)
    preds = np.vstack([np.asarray(p1)[order], np.asarray(p2)[order]])
    aucs, cov = _fast_delong(preds, int(y.sum()))
    var = cov[0, 0] + cov[1, 1] - 2 * cov[0, 1]
    z = (aucs[0] - aucs[1]) / np.sqrt(var)
    return float(aucs[0]), float(aucs[1]), float(z), float(2 * stats.norm.sf(abs(z)))


def paired_boot_delta(y, pa, pb, B=C.N_BOOT, seed=C.SEED):
    """Paired bootstrap of AUC(a) - AUC(b); percentile 95% CI."""
    y = np.asarray(y); pa = np.asarray(pa); pb = np.asarray(pb); n = len(y)
    rng = np.random.default_rng(seed); d = np.empty(B)
    for b in range(B):
        i = rng.integers(0, n, n)
        d[b] = auc(y[i], pa[i]) - auc(y[i], pb[i])
    return float(auc(y, pa) - auc(y, pb)), float(np.quantile(d, 0.025)), float(np.quantile(d, 0.975))


# ---------- reclassification ----------
def _nri_idi(y, p_old, p_new, cuts):
    y = np.asarray(y).astype(bool)
    co, cn = np.digitize(p_old, cuts), np.digitize(p_new, cuts)
    up, dn = cn > co, cn < co
    nri_cat = (up[y].mean() - dn[y].mean()) + (dn[~y].mean() - up[~y].mean())
    d = np.sign(p_new - p_old)
    nri_cont = ((d[y] > 0).mean() - (d[y] < 0).mean()) + ((d[~y] < 0).mean() - (d[~y] > 0).mean())
    idi = (p_new[y].mean() - p_old[y].mean()) - (p_new[~y].mean() - p_old[~y].mean())
    return np.array([nri_cat, nri_cont, idi])


def reclassification(y, p_old, p_new, cuts=C.NRI_CUTS, B=C.N_BOOT, seed=C.SEED):
    y = np.asarray(y); p_old = np.asarray(p_old); p_new = np.asarray(p_new); n = len(y)
    est = _nri_idi(y, p_old, p_new, cuts)
    rng = np.random.default_rng(seed); bs = np.empty((B, 3))
    for b in range(B):
        i = rng.integers(0, n, n)
        bs[b] = _nri_idi(y[i], p_old[i], p_new[i], cuts)
    lo, hi = np.quantile(bs, 0.025, axis=0), np.quantile(bs, 0.975, axis=0)
    return {k: (float(est[j]), float(lo[j]), float(hi[j])) for j, k in enumerate(["nri_cat", "nri_cont", "idi"])}


# ---------- calibration ----------
def calibration(y, p, eps=1e-6):
    y = np.asarray(y); p = np.clip(np.asarray(p), eps, 1 - eps)
    lg = np.log(p / (1 - p))
    slope = sm.GLM(y, sm.add_constant(lg), family=sm.families.Binomial()).fit().params[1]
    citl = sm.GLM(y, np.ones((len(y), 1)), family=sm.families.Binomial(), offset=lg).fit().params[0]
    prev = y.mean()
    return dict(brier=float(np.mean((p - y) ** 2)), brier_null=float(prev * (1 - prev)),
                slope=float(slope), intercept=float(citl))


def calibration_curve(y, p, bins=10):
    y = np.asarray(y); p = np.asarray(p)
    edges = np.quantile(p, np.linspace(0, 1, bins + 1)); edges[-1] += 1e-9
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, bins - 1)
    return np.array([[p[idx == b].mean(), y[idx == b].mean()] for b in range(bins) if (idx == b).any()])


# ---------- decision curve ----------
def net_benefit(y, p, thresholds=C.DCA_THRESHOLDS):
    y = np.asarray(y); n = len(y); out = []
    for t in thresholds:
        pred = p >= t
        tp = (pred & (y == 1)).sum(); fp = (pred & (y == 0)).sum()
        out.append(tp / n - fp / n * t / (1 - t))
    return np.array(out)


def treat_all(y, thresholds=C.DCA_THRESHOLDS):
    y = np.asarray(y); prev = y.mean()
    return np.array([prev - (1 - prev) * t / (1 - t) for t in thresholds])
