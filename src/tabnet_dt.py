"""TabNet-family models re-implemented from first principles in PyTorch (GLU feature transformers,
sparsemax attentive transformers, prior-scale update) and the DT-XTabNet dual-task architecture with
Attention Divergence Regularisation (ADR)."""
import copy, math
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from sklearn.model_selection import train_test_split
from . import config as C
from .stats_utils import auc

SQ = math.sqrt(0.5)


def sparsemax(z):
    zs, _ = torch.sort(z, dim=-1, descending=True)
    k = torch.arange(1, z.size(-1) + 1, device=z.device, dtype=z.dtype)
    cs = zs.cumsum(-1)
    support = (1 + k * zs) > cs
    kz = support.sum(-1, keepdim=True)
    tau = (cs.gather(-1, kz - 1) - 1) / kz.to(z.dtype)
    return torch.clamp(z - tau, min=0)


class GBN(nn.Module):
    def __init__(self, dim, vbs=32, momentum=0.02):
        super().__init__(); self.bn = nn.BatchNorm1d(dim, momentum=momentum); self.vbs = vbs

    def forward(self, x):
        if self.training and x.size(0) > self.vbs and x.size(0) % self.vbs == 0:
            return torch.cat([self.bn(c) for c in x.chunk(x.size(0) // self.vbs, 0)], 0)
        return self.bn(x)


class GLU(nn.Module):
    def __init__(self, i, o, vbs):
        super().__init__(); self.fc = nn.Linear(i, 2 * o, bias=False); self.bn = GBN(2 * o, vbs)

    def forward(self, x):
        return F.glu(self.bn(self.fc(x)))


class FeatureTransformer(nn.Module):
    def __init__(self, shared, dim, vbs, n_ind=2):
        super().__init__(); self.shared = shared
        self.ind = nn.ModuleList([GLU(dim, dim, vbs) for _ in range(n_ind)])

    def forward(self, x):
        x = self.shared[0](x)
        for b in self.shared[1:]:
            x = (x + b(x)) * SQ
        for b in self.ind:
            x = (x + b(x)) * SQ
        return x


class Attentive(nn.Module):
    def __init__(self, n_a, n_feat, vbs):
        super().__init__(); self.fc = nn.Linear(n_a, n_feat, bias=False); self.bn = GBN(n_feat, vbs)

    def forward(self, a, prior):
        return sparsemax(prior * self.bn(self.fc(a)))


class TabNetFamily(nn.Module):
    """topology: 'dt'     -> shared trunk (2 steps) + two task branches (3 steps each), one head per branch (DT-XTabNet)
                 'single' -> one 5-step TabNet, one head (one model of the 'two independent TabNets' baseline)
                 'shared' -> one 5-step TabNet, two heads on the same decision embedding (Shared-Trunk TabNet)"""
    def __init__(self, n_cont, cat_card, topology="dt", n_d=32, n_a=32, gamma=1.5, emb_dim=4, dropout=0.1,
                 vbs=32, n_shared_steps=2, n_branch_steps=3):
        super().__init__()
        self.topology, self.n_d, self.n_a, self.gamma, self.emb_dim = topology, n_d, n_a, gamma, emb_dim
        self.n_cont, self.n_cat = n_cont, len(cat_card); self.n_feat = n_cont + self.n_cat
        d_in = n_cont + emb_dim * self.n_cat; dim = n_d + n_a
        self.emb = nn.ModuleList([nn.Embedding(k, emb_dim) for k in cat_card])
        self.bn_in = nn.BatchNorm1d(n_cont)
        n_trunk = n_shared_steps if topology == "dt" else n_shared_steps + n_branch_steps
        self.n_trunk = n_trunk; self.n_branch_steps = n_branch_steps if topology == "dt" else 0
        mk_shared = lambda: nn.ModuleList([GLU(d_in, dim, vbs), GLU(dim, dim, vbs)])
        shared = mk_shared()
        self.ft0 = FeatureTransformer(shared, dim, vbs)
        self.att = nn.ModuleList([Attentive(n_a, self.n_feat, vbs) for _ in range(n_trunk)])
        self.ft = nn.ModuleList([FeatureTransformer(shared, dim, vbs) for _ in range(n_trunk)])
        self.branches = nn.ModuleList()
        if topology == "dt":
            for _ in range(2):
                sb = mk_shared()
                self.branches.append(nn.ModuleDict(dict(
                    att=nn.ModuleList([Attentive(n_a, self.n_feat, vbs) for _ in range(self.n_branch_steps)]),
                    ft=nn.ModuleList([FeatureTransformer(sb, dim, vbs) for _ in range(self.n_branch_steps)]))))
        self.n_heads = 1 if topology == "single" else 2
        self.drop = nn.Dropout(dropout)
        self.heads = nn.ModuleList([nn.Linear(n_d, 1) for _ in range(self.n_heads)])

    def _mask_input(self, cont, embs, m):
        parts = [cont * m[:, :self.n_cont]]
        if self.n_cat:
            parts.append((embs * m[:, self.n_cont:, None]).flatten(1))
        return torch.cat(parts, 1)

    def _run(self, cont, embs, att, ft, a, prior, dsum, masks):
        for at, f in zip(att, ft):
            m = at(a, prior); prior = prior * (self.gamma - m)
            out = f(self._mask_input(cont, embs, m))
            dsum = dsum + F.relu(out[:, :self.n_d]); a = out[:, self.n_d:]; masks.append(m)
        return a, prior, dsum

    def forward(self, cont, cat):
        cont = self.bn_in(cont)
        embs = torch.stack([e(cat[:, j]) for j, e in enumerate(self.emb)], 1) if self.n_cat else None
        B = cont.size(0)
        ones = torch.ones(B, self.n_feat, device=cont.device)
        a = self.ft0(self._mask_input(cont, embs, ones))[:, self.n_d:]
        prior = ones; dsum = torch.zeros(B, self.n_d, device=cont.device); masks = []
        a, prior, dsum = self._run(cont, embs, self.att, self.ft, a, prior, dsum, masks)
        if self.topology == "dt":
            logits, aggs = [], []
            for br, head in zip(self.branches, self.heads):
                bm = []
                _, _, d = self._run(cont, embs, br["att"], br["ft"], a, prior, dsum, bm)
                logits.append(head(self.drop(d))); aggs.append(torch.stack(bm).mean(0))
            return torch.cat(logits, 1), aggs
        agg = torch.stack(masks).mean(0)
        return torch.cat([h(self.drop(dsum)) for h in self.heads], 1), [agg]


class SharedBottomMLP(nn.Module):
    def __init__(self, d_in, h1=128, h2=64, dropout=0.1):
        super().__init__()
        self.body = nn.Sequential(nn.Linear(d_in, h1), nn.ReLU(), nn.Dropout(dropout), nn.Linear(h1, h2), nn.ReLU())
        self.heads = nn.ModuleList([nn.Linear(h2, 1), nn.Linear(h2, 1)])

    def forward(self, x, _cat=None):
        h = self.body(x); return torch.cat([hd(h) for hd in self.heads], 1), []


def n_params(m):
    return int(sum(p.numel() for p in m.parameters() if p.requires_grad))


def adr_penalty(aggs):
    return F.cosine_similarity(aggs[0], aggs[1], dim=1).mean() if len(aggs) == 2 else torch.zeros(())


def fit_predict(make_model, Xc_tr, Xk_tr, Y_tr, Xc_te, Xk_te, lam=0.0, seed=C.SEED, lr=None, cfg=C.TABNET,
                strat=None, mlp=False):
    """Train with early stopping on an internal 15% split; return test probabilities, mean test cosine, model."""
    torch.manual_seed(seed); np.random.seed(seed); torch.set_num_threads(1)
    i_tr, i_va = train_test_split(np.arange(len(Y_tr)), test_size=cfg["val_frac"], random_state=seed, stratify=strat)
    T = lambda a, dt=torch.float32: torch.as_tensor(a, dtype=dt)
    xc, xk, y = T(Xc_tr), T(Xk_tr, torch.long), T(Y_tr)
    model = make_model(); opt = torch.optim.Adam(model.parameters(), lr=lr or cfg["lr"], weight_decay=cfg["wd"])
    g = torch.Generator().manual_seed(seed); best, best_state, bad = 1e9, None, 0

    def loss_fn(logits, yy, aggs):
        return F.binary_cross_entropy_with_logits(logits, yy, reduction="none").mean(0).sum()

    for epoch in range(cfg["max_epochs"]):
        model.train(); perm = i_tr[torch.randperm(len(i_tr), generator=g).numpy()]
        for s in range(0, len(perm) - cfg["batch"] + 1, cfg["batch"]):
            b = perm[s:s + cfg["batch"]]
            logits, aggs = model(xc[b], xk[b]); loss = loss_fn(logits, y[b], aggs)
            if lam > 0 and len(aggs) == 2:
                loss = loss + lam * adr_penalty(aggs)
            opt.zero_grad(); loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            lg, ag = model(xc[i_va], xk[i_va]); vl = loss_fn(lg, y[i_va], ag).item()
        if vl < best - 1e-5:
            best, best_state, bad = vl, copy.deepcopy(model.state_dict()), 0
        else:
            bad += 1
            if bad >= cfg["patience"]:
                break
    model.load_state_dict(best_state); model.eval()
    with torch.no_grad():
        lg, ag = model(T(Xc_te), T(Xk_te, torch.long))
    cos = float(adr_penalty(ag)) if len(ag) == 2 else float("nan")
    return torch.sigmoid(lg).numpy(), cos, model, epoch + 1
