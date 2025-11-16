
from __future__ import annotations

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler


def make_preprocessor(strategy: str = "median", scale: str = "standard") -> tuple[SimpleImputer, StandardScaler | None]:
    """Construct imputer and scaler according to config."""
    imputer = SimpleImputer(strategy=strategy)
    scaler = StandardScaler() if scale == "standard" else None
    return imputer, scaler
