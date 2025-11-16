
from __future__ import annotations

from typing import Any, Dict, Iterable, Optional, Tuple

import numpy as np
import xgboost as xgb

from ..metrics import c_index


def fit_xgb_cox_hpo(
    X_train: np.ndarray,
    ytime_train: np.ndarray,
    yevent_train: np.ndarray,
    X_val: np.ndarray,
    ytime_val: np.ndarray,
    yevent_val: np.ndarray,
    search_space: Optional[Dict[str, Iterable[Any]]] = None,
    n_trials: int = 20,
    random_state: int = 42,
) -> Dict[str, Any]:
    """Tiny random search over XGBoost-Cox hyperparameters; returns the best booster and params."""
    rng = np.random.RandomState(random_state)
    space = search_space or {
        "max_depth": [2, 3, 4, 5, 6],
        "learning_rate": [0.01, 0.03, 0.05, 0.1],
        "subsample": [0.6, 0.8, 1.0],
        "colsample_bytree": [0.6, 0.8, 1.0],
        "reg_lambda": [0.0, 0.5, 1.0, 2.0],
        "reg_alpha": [0.0, 0.1, 0.5],
        "n_estimators": [200, 400, 600, 800],
    }

    def sample() -> Dict[str, Any]:
        return {k: rng.choice(v) for k, v in space.items()}

    def build_params(p: Dict[str, Any]) -> Tuple[Dict[str, Any], int]:
        base = dict(objective="survival:cox", eval_metric="cox-nloglik", tree_method="hist")
        base.update({k: v for k, v in p.items() if k != "n_estimators"})
        return base, int(p.get("n_estimators", 400))

    best = {"cindex": -1.0, "params": None, "n_estimators": None, "booster": None}

    dtrain = xgb.DMatrix(X_train, label=np.where(yevent_train == 1, ytime_train, -ytime_train))
    dval = xgb.DMatrix(X_val, label=np.where(yevent_val == 1, ytime_val, -ytime_val))

    for _ in range(int(n_trials)):
        p = sample()
        params, n_est = build_params(p)
        bst = xgb.train(params, dtrain, num_boost_round=n_est, verbose_eval=False)
        lp_val = bst.predict(dval)
        ci = c_index(ytime_val, yevent_val, lp_val)
        if ci > best["cindex"]:
            best = {"cindex": float(ci), "params": params, "n_estimators": int(n_est), "booster": bst}

    return best


def predict_lp(booster: xgb.Booster, X: np.ndarray) -> np.ndarray:
    """Predict linear predictor for X using an XGBoost Booster."""
    return booster.predict(xgb.DMatrix(X))
