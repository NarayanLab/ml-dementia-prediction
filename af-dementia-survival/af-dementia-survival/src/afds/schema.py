
from __future__ import annotations

import pandas as pd


def validate_schema(df: pd.DataFrame, id_col: str | None, time_col: str, event_col: str) -> None:
    """Validate required columns and basic invariants.

    - `time` must be present and non-negative
    - `event` must be present and in {0, 1}
    """
    missing = [c for c in [time_col, event_col] if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    if (df[time_col] < 0).any():
        raise ValueError("Negative follow-up times detected.")

    if (~df[event_col].isin([0, 1])).any():
        raise ValueError("Event column must be 0/1.")
