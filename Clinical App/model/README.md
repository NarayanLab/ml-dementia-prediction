# Production model (22-feature XGBoost-Cox)

The model the backend serves. `MODEL_DIR` in `backend/main.py` points here.

| File | Role |
|---|---|
| `xgb_cox_22feature.json` | **The model.** 22 features, `survival:cox`, 141 trees |
| `meta.json` | The 22 feature names, **in column order** |
| `feature_means.json` | Population means — frontend defaults + Calcium fill value |
| `baseline.json` | `S0(t*=5y) = 0.9701937` |
| `dca_band.json` | Risk thresholds `[0.02, 0.04481]` |
| `training_meta.json` | Hyperparameters, selection method, held-out C-index |
| `feature_importance_ranking.json` | Importance **ranking**. NOT the column order — see below |

## Provenance

Features selected by BIC, then refit using the 77-feature parent model's best
hyperparameters (`"selection": "BIC_features_previous_best_hparams"`,
`"source_hparams": "previous_77_feature_model"`). Held-out **C-index 0.822**.

Trained with XGBoost 3.0.3 using `gpu_hist`; loads and predicts correctly under
the 2.1.3 pinned in `backend/requirements.txt` on CPU, so that pin does not need
bumping.

The 77-feature parent model and the study data dictionary are deliberately **not
in version control** — they live alongside this repo in the lab's OneDrive. The
encodings that matter for serving are reproduced below so this folder is
self-sufficient.

## Traps

**Column order comes from the booster, not from a file.** `main.py` uses
`booster.feature_names`. `feature_importance_ranking.json` is an importance
ranking and disagrees with the true column order — using it would silently
scramble every input and produce plausible-looking wrong risks.
`backend/test_backend.py` asserts both facts.

**Baseline and thresholds belong to this fit.** `S0_tstar` and the DCA band are
Breslow/DCA quantities derived from this specific model. The 77-feature parent
carries different values (0.9696292, high band 0.048354). Mixing a booster from
one fit with the baseline from another mis-scales every reported risk without any
error surfacing.

**QRS is axis in degrees, not duration in ms.** Range roughly -80 to +195; the
study data dictionary lists the variable as "QRS-Axis". The UI label was
corrected in commits `e308125` / `25c358e`.

**Do not judge calibration by feeding in the mean patient.** `exp` is convex, so
by Jensen's inequality the patient at the mean covariate vector sits *below*
population average risk (here ~1.9% against a 2.98% baseline). That is expected,
not a bug. Validate with the held-out C-index instead.

## Encodings

- `Marital` — 0 Single, 1 Married, 2 Divorced/Widowed, 3 Unknown
- `Insurance` — 0 Public, 1 Private, 2 Unknown
- `AF_Age` — continuous, in years (the superseded 20-feature model used
  `AF_age_2/3/4` band dummies, which made predicted risk flat below 65)

`Calcium_missing` and `HCT_missing` are the only two missingness flags selected
into the 22, which is why the form surfaces availability toggles for exactly
those two labs and no others. When Calcium is marked unavailable the value is
filled from `feature_means.json` and the flag set to 1, matching training.

## Known quirk

PPI slightly *lowers* predicted risk (4.25% vs 4.47% at the age-80 reference).
Small, plausibly noise in a depth-3 tree, but worth a look if PPI was selected on
the hypothesis of a positive association.

## Superseded model

The 20-feature model this replaced (`xgb_cox_model.json`, `baseline_hazard.json`,
`feature_manifest.json`, `app_metadata.json`) was removed from the working tree
but remains in history:

```bash
git show 33522b2:"Clinical App/xgb_cox_model.json" > old_model.json
```
