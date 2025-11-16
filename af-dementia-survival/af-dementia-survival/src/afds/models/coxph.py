
from __future__ import annotations

from typing import Sequence

import pandas as pd
from lifelines import CoxPHFitter


def fit_coxph(df: pd.DataFrame, time_col: str, event_col: str, feature_cols: Sequence[str]) -> CoxPHFitter:
    """Fit a Cox proportional hazards model with lifelines."""
    data = df[[time_col, event_col] + list(feature_cols)].copy()
    cph = CoxPHFitter()
    cph.fit(data, duration_col=time_col, event_col=event_col, show_progress=False)
    return cph
