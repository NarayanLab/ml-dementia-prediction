
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lifelines import KaplanMeierFitter


def _ensure_outdir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def calibration_plot(preds_csv: str | Path, horizon: float, outdir: str | Path, n_bins: int = 10) -> Path:
    """Binned calibration: predicted risk vs observed KM risk at a horizon."""
    preds_csv = Path(preds_csv)
    outdir = _ensure_outdir(Path(outdir))

    df = pd.read_csv(preds_csv)
    if "risk_at_horizon" not in df.columns:
        raise ValueError("preds csv must contain a 'risk_at_horizon' column")

    df = df.copy()
    df["bin"] = pd.qcut(df["risk_at_horizon"], q=n_bins, duplicates="drop")

    rows = []
    for _, g in df.groupby("bin", observed=False):
        kmf = KaplanMeierFitter()
        kmf.fit(g["time"].values, event_observed=g["event"].values)
        surv = kmf.survival_function_at_times(horizon).values[0]
        rows.append({"pred": g["risk_at_horizon"].mean(), "obs": 1.0 - float(surv)})

    cal = pd.DataFrame(rows).sort_values("pred")

    fig, ax = plt.subplots()
    ax.scatter(cal["pred"], cal["obs"], s=18)
    mx = float(max(cal["pred"].max(), cal["obs"].max()))
    ax.plot([0, mx], [0, mx], linestyle="--", linewidth=1)
    ax.set_xlabel("Predicted risk @ horizon")
    ax.set_ylabel("Observed risk (KM) @ horizon")
    ax.set_title(f"Calibration (n_bins={n_bins}, horizon={horizon:g})")

    outpath = outdir / "calibration.png"
    fig.savefig(outpath, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return outpath


def km_by_thresholds(preds_csv: str | Path, thresholds: Iterable[float], outdir: str | Path) -> Path:
    """KM curves stratified by risk thresholds (low/intermediate/high)."""
    preds_csv = Path(preds_csv)
    outdir = _ensure_outdir(Path(outdir))

    df = pd.read_csv(preds_csv)
    if "risk_at_horizon" not in df.columns:
        raise ValueError("preds csv must contain a 'risk_at_horizon' column")

    th = sorted([float(t) for t in thresholds])

    def assign(p: float) -> str:
        return "low" if p < th[0] else ("intermediate" if p < th[1] else "high")

    df = df.copy()
    df["group"] = [assign(p) for p in df["risk_at_horizon"].values]

    fig, ax = plt.subplots()
    for grp, sub in df.groupby("group"):
        kmf = KaplanMeierFitter(label=grp)
        kmf.fit(durations=sub["time"].values, event_observed=sub["event"].values)
        kmf.plot(ax=ax)

    ax.set_title(f"KM by decision thresholds {th}")
    ax.set_xlabel("Days")
    ax.set_ylabel("Survival probability")

    outpath = outdir / "km_by_thresholds.png"
    fig.savefig(outpath, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return outpath
