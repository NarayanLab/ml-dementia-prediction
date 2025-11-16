
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional

import pandas as pd
import yaml
import json


def load_csv(path: str | Path) -> pd.DataFrame:
    """Load a CSV into a DataFrame."""
    return pd.read_csv(path)


def save_json(obj: dict, path: str | Path) -> None:
    """Save an object as pretty JSON."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def load_config(path: str | Path) -> dict:
    """Load a YAML config file."""
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def infer_features(
    df: pd.DataFrame,
    id_col: Optional[str],
    time_col: str,
    event_col: str,
    include: Optional[Iterable[str]] = None,
    exclude: Optional[Iterable[str]] = None,
) -> list[str]:
    """Return model feature columns.

    Preference order:
    1. If `include` is provided, return intersection of `include` with df columns.
    2. Otherwise, return all numeric columns excluding id/time/event and any in `exclude`.
    """
    include = list(include or [])
    excl = set(exclude or []) | {time_col, event_col}
    if id_col:
        excl.add(id_col)

    if include:
        return [c for c in include if c in df.columns]

    return [c for c in df.columns if c not in excl and pd.api.types.is_numeric_dtype(df[c])]
