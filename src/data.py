"""Data loading, panel construction and fold-safe preprocessing."""
import numpy as np, pandas as pd
from pandas.api.types import is_numeric_dtype
from sklearn.model_selection import StratifiedKFold
from . import config as C


def load(path=None):
    df = pd.read_excel(path or C.DATA_PATH, sheet_name=0, header=2)   # row 3 holds variable names
    assert df.shape[0] == 1500, f"expected 1,500 records, found {df.shape[0]}"
    return df


def outcomes(df):
    ckd = (df["CKD_Stage"] >= C.HIGH_RISK_STAGE).astype(int).values
    cvd = (df["CVD_Stage"] >= C.HIGH_RISK_STAGE).astype(int).values
    return ckd, cvd


def predictor_columns(df):
    """All columns except ID and the three outcome labels; constant columns are dropped and reported."""
    cols = [c for c in df.columns if c != C.ID_COL and c not in C.LABELS]
    dropped = [c for c in cols if df[c].nunique(dropna=False) <= 1]
    return [c for c in cols if c not in dropped], dropped


def panels(df):
    full, dropped = predictor_columns(df)
    nonrenal = [c for c in full if c not in C.RENAL]
    return dict(full=full, nonrenal=nonrenal, curated=C.CURATED, agesex=C.AGE_SEX), dropped


def is_categorical(df, c):
    return not is_numeric_dtype(df[c])


def folds(df):
    ckd, cvd = outcomes(df)
    strat = ckd * 2 + cvd                       # joint CKD-high x CVD-high stratification
    skf = StratifiedKFold(C.N_SPLITS, shuffle=True, random_state=C.SEED)
    return list(skf.split(np.zeros(len(df)), strat)), strat


class Preprocessor:
    """log1p (fixed list) -> z-score (training-fold statistics) ; categoricals integer-coded (0 = unknown)."""
    def fit(self, df, cols):
        self.cols = list(cols)
        self.cat = [c for c in self.cols if not is_numeric_dtype(df[c])]
        self.num = [c for c in self.cols if c not in self.cat]
        t = self._log(df)
        self.mu = t[self.num].mean()
        self.sd = t[self.num].std(ddof=0).replace(0, 1.0)
        self.maps = {c: {v: i + 1 for i, v in enumerate(sorted(df[c].dropna().unique()))} for c in self.cat}
        return self

    def _log(self, df):
        t = df[self.cols].copy()
        for c in self.num:
            if c in C.LOG1P_VARS:
                t[c] = np.log1p(t[c])
        return t

    def cont(self, df):
        t = self._log(df)
        return ((t[self.num] - self.mu) / self.sd).values.astype(np.float32)

    def codes(self, df):
        if not self.cat:
            return np.zeros((len(df), 0), dtype=np.int64)
        return np.stack([df[c].map(self.maps[c]).fillna(0).astype(int).values for c in self.cat], 1)

    def onehot(self, df):
        parts = [self.cont(df)]
        codes = self.codes(df)
        for j, c in enumerate(self.cat):
            k = len(self.maps[c])
            parts.append(np.eye(k + 1, dtype=np.float32)[codes[:, j]][:, 1:])
        return np.hstack(parts)

    @property
    def cat_cardinalities(self):
        return [len(self.maps[c]) + 1 for c in self.cat]
