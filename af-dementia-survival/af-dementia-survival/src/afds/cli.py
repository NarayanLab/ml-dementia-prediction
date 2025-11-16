
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import typer

from .io import infer_features, load_config, load_csv, save_json
from .metrics import breslow_baseline, c_index, survival_at_horizon
from .models.coxph import fit_coxph
from .models.deepsurv import fit_deepsurv_hpo
from .models.xgboost_cox import fit_xgb_cox_hpo, predict_lp as xgb_predict_lp
from .preprocess import make_preprocessor
from .schema import validate_schema

app = typer.Typer(help="AF Dementia Survival CLI")


def _prep(split_dir: str | Path, cfg: dict):
    sd = Path(split_dir)
    train = load_csv(sd / "train.csv")
    val = load_csv(sd / "val.csv")
    test = load_csv(sd / "test.csv")

    id_col = cfg["columns"].get("id")
    time_col = cfg["columns"]["time"]
    event_col = cfg["columns"]["event"]
    feats = infer_features(
        train,
        id_col=id_col,
        time_col=time_col,
        event_col=event_col,
        include=cfg["features"]["include"],
        exclude=cfg["features"]["exclude"],
    )
    imputer, scaler = make_preprocessor(cfg["preprocess"]["impute"], cfg["preprocess"]["scale"])

    Xtr = pd.DataFrame(imputer.fit_transform(train[feats]), columns=feats, index=train.index)
    Xva = pd.DataFrame(imputer.transform(val[feats]), columns=feats, index=val.index)
    Xte = pd.DataFrame(imputer.transform(test[feats]), columns=feats, index=test.index)

    ytr_t, ytr_e = train[time_col].values, train[event_col].values
    yva_t, yva_e = val[time_col].values, val[event_col].values
    yte_t, yte_e = test[time_col].values, test[event_col].values

    return (train, val, test, feats, Xtr, Xva, Xte, ytr_t, ytr_e, yva_t, yva_e, yte_t, yte_e, id_col, time_col, event_col)


@app.command()
def validate(csv: str, config: str):
    """Validate required columns/types in CSV using the given config."""
    cfg = load_config(config)
    df = load_csv(csv)
    validate_schema(df, cfg["columns"].get("id"), cfg["columns"]["time"], cfg["columns"]["event"])
    typer.echo(f"Schema OK: {df.shape}")


@app.command()
def split(csv: str, out: str, seed: int = 42):
    """Split a single CSV into train/val/test with 60/20/20 proportions."""
    outp = Path(out)
    outp.mkdir(parents=True, exist_ok=True)
    df = load_csv(csv)
    m = np.random.RandomState(seed).rand(len(df))
    tr = df[m < 0.6]
    va = df[(m >= 0.6) & (m < 0.8)]
    te = df[m >= 0.8]
    tr.to_csv(outp / "train.csv", index=False)
    va.to_csv(outp / "val.csv", index=False)
    te.to_csv(outp / "test.csv", index=False)
    typer.echo("Wrote splits to " + str(outp))


@app.command("train")
def train(
    model: str,
    split_dir: str,
    config: str,
    out: str,
    hpo_trials: int = 20,
    horizon: float = 3600.0,
):
    """Train a model (coxph | xgb-cox | deepsurv) and write predictions/metrics."""
    cfg = load_config(config)
    outp = Path(out)
    outp.mkdir(parents=True, exist_ok=True)

    (
        train,
        val,
        test,
        feats,
        Xtr,
        Xva,
        Xte,
        ytr_t,
        ytr_e,
        yva_t,
        yva_e,
        yte_t,
        yte_e,
        id_col,
        time_col,
        event_col,
    ) = _prep(split_dir, cfg)

    if model.lower() == "coxph":
        dftr = pd.concat([train[[time_col, event_col]], Xtr], axis=1)
        cph = fit_coxph(dftr, time_col, event_col, feats)
        lp_tr = cph.predict_partial_hazard(Xtr).values.ravel()
        lp_va = cph.predict_partial_hazard(Xva).values.ravel()
        lp_te = cph.predict_partial_hazard(Xte).values.ravel()
        base = cph.baseline_cumulative_hazard_.reset_index()
        base.columns = ["time", "cum_hazard"]
        cph.save_model(outp / "model.cox.json")
        base.to_json(outp / "baseline_h0.json", orient="records")
        save_json({"features": feats, "model": "coxph"}, outp / "feature_manifest.json")

    elif model.lower() == "xgb-cox":
        best = fit_xgb_cox_hpo(Xtr.values, ytr_t, ytr_e, Xva.values, yva_t, yva_e, n_trials=int(hpo_trials))
        bst = best["booster"]
        lp_tr = xgb_predict_lp(bst, Xtr.values)
        lp_va = xgb_predict_lp(bst, Xva.values)
        lp_te = xgb_predict_lp(bst, Xte.values)
        base = breslow_baseline(ytr_t, ytr_e, lp_tr)
        bst.save_model(outp / "model.xgb.json")
        base.to_json(outp / "baseline_h0.json", orient="records")
        save_json(
            {"features": feats, "model": "xgb-cox", "best_params": best["params"], "n_estimators": best["n_estimators"]},
            outp / "feature_manifest.json",
        )

    elif model.lower() == "deepsurv":
        best = fit_deepsurv_hpo(Xtr.values, ytr_t, ytr_e, Xva.values, yva_t, yva_e, n_trials=int(hpo_trials))
        import torch as _torch

        model_obj = best["model"]
        _torch.save(model_obj.state_dict(), outp / "model.deepsurv.pt")
        lp_tr = model_obj(_torch.tensor(Xtr.values, dtype=_torch.float32)).detach().cpu().numpy()
        lp_va = model_obj(_torch.tensor(Xva.values, dtype=_torch.float32)).detach().cpu().numpy()
        lp_te = model_obj(_torch.tensor(Xte.values, dtype=_torch.float32)).detach().cpu().numpy()
        base = breslow_baseline(ytr_t, ytr_e, lp_tr)
        base.to_json(outp / "baseline_h0.json", orient="records")
        save_json({"features": feats, "model": "deepsurv", "best_params": best["params"]}, outp / "feature_manifest.json")

    else:
        raise typer.BadParameter("Model must be one of: coxph, xgb-cox, deepsurv")

    def _save(split_name: str, df_raw: pd.DataFrame, lp, t, e) -> float:
        dfp = pd.DataFrame({"time": t, "event": e, "lp": lp})
        base_df = pd.read_json(outp / "baseline_h0.json")
        dfp["risk_at_horizon"] = 1 - survival_at_horizon(base_df, dfp["lp"].values, horizon)
        dfp.to_csv(outp / f"pred_{split_name}.csv", index=False)
        return c_index(t, e, lp)

    ci_tr = _save("train", train, lp_tr, ytr_t, ytr_e)
    ci_va = _save("val", val, lp_va, yva_t, yva_e)
    ci_te = _save("test", test, lp_te, yte_t, yte_e)

    save_json({"cindex": {"train": ci_tr, "val": ci_va, "test": ci_te}, "horizon": horizon}, outp / "metrics.json")
    typer.echo(f"Done. C-index: train={ci_tr:.3f} val={ci_va:.3f} test={ci_te:.3f}")


@app.command("plot")
def plot(
    run: str,
    what: str = "calibration,km",
    horizon: float = 3600.0,
    thresholds: str = "0.02,0.07",
    out: Optional[str] = None,
):
    """Generate calibration and KM-by-thresholds figures from a run directory."""
    from .plots import calibration_plot, km_by_thresholds

    runp = Path(run)
    outp = Path(out) if out else (runp / "figs")
    outp.mkdir(parents=True, exist_ok=True)

    preds_csv = runp / "pred_test.csv"

    if "calibration" in what:
        calibration_plot(preds_csv, horizon=horizon, outdir=outp)

    if "km" in what:
        th = [float(x.strip()) for x in thresholds.split(",") if x.strip()]
        km_by_thresholds(preds_csv, thresholds=th, outdir=outp)

    typer.echo(f"Saved plots to {outp}")


if __name__ == "__main__":
    app()
