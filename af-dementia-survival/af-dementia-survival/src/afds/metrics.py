
from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd
from lifelines.utils import concordance_index


def c_index(time: Iterable[float], event: Iterable[int], risk: Iterable[float]) -> float:
    """Harrell's C-index (higher is better)."""
    return concordance_index(time, -np.asarray(risk), event)


def breslow_baseline(time, event, linpred) -> pd.DataFrame:
    """Compute Breslow baseline cumulative hazard from linear predictors."""
    time = np.asarray(time)
    event = np.asarray(event).astype(int)
    lp = np.asarray(linpred)

    order = np.argsort(time)
    t = time[order]
    e = event[order]
    r = np.exp(lp[order])

    uniq_t, first_idx = np.unique(t, return_index=True)
    rev_csum = np.cumsum(r[::-1])[::-1]
    denom = rev_csum[first_idx]
    d_j = np.add.reduceat(e, first_idx)
    dH = np.where(denom > 0, d_j / denom, 0.0)
    H0 = np.cumsum(dH)

    return pd.DataFrame({"time": uniq_t, "cum_hazard": H0})


def survival_at_horizon(h0_df: pd.DataFrame, lp: np.ndarray, horizon: float) -> np.ndarray:
    """S(h | x) given baseline cumulative hazard and linear predictor."""
    if horizon <= h0_df["time"].iloc[0]:
        Ht = float(h0_df["cum_hazard"].iloc[0])
    else:
        Ht = float(np.interp(horizon, h0_df["time"].values, h0_df["cum_hazard"].values))
    return np.exp(-Ht * np.exp(lp))
