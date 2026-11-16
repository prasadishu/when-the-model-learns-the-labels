"""Classical single-task models and the cross-validation driver (library defaults, seed 42)."""
import numpy as np, warnings
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from . import config as C
from .data import Preprocessor
from .stats_utils import auc

warnings.filterwarnings("ignore")

CLASSICAL = {
    "Logistic Regression": lambda: LogisticRegression(max_iter=5000),
    "Random Forest": lambda: RandomForestClassifier(n_estimators=500, random_state=C.SEED, n_jobs=1),
    "SVM (RBF)": lambda: SVC(kernel="rbf", probability=True, random_state=C.SEED),
    "Gradient Boosting": lambda: GradientBoostingClassifier(random_state=C.SEED),
    "XGBoost": lambda: XGBClassifier(random_state=C.SEED, n_jobs=1, verbosity=0, eval_metric="logloss"),
    "LightGBM": lambda: LGBMClassifier(random_state=C.SEED, n_jobs=1, verbose=-1),
}


def cv_oof(model_name, df, cols, y, fold_idx):
    """Out-of-fold probabilities and per-fold AUCs; preprocessing is refit inside each training fold."""
    oof = np.zeros(len(df)); fold_auc = []; last = None
    for tr, te in fold_idx:
        pp = Preprocessor().fit(df.iloc[tr], cols)
        m = CLASSICAL[model_name]()
        m.fit(pp.onehot(df.iloc[tr]), y[tr])
        oof[te] = m.predict_proba(pp.onehot(df.iloc[te]))[:, 1]
        fold_auc.append(auc(y[te], oof[te])); last = (m, pp)
    return oof, np.array(fold_auc), last


def feature_names(pp):
    names = list(pp.num)
    for c in pp.cat:
        names += [f"{c}={v}" for v in sorted(pp.maps[c], key=pp.maps[c].get)]
    return names
