
# AF Dementia Survival

A polished, readable codebase for time-to-event modeling in AF cohorts (CoxPH, XGBoost-Cox, DeepSurv).
Includes a fully numeric example dataset and a lookup table for categorical encodings.

## Quickstart

```bash
conda env create -f environment.yml
conda activate af-dementia-survival
pip install -e .
```

## Example workflow
```bash
# Validate schema & split
afds validate --csv data/example/example_cohort.csv --config configs/default.yaml
afds split    --csv data/example/example_cohort.csv --out data/splits --seed 42

# Train
afds train coxph    --split-dir data/splits --config configs/default.yaml --out runs/cox --horizon 3600
# Optional:
# afds train xgb-cox  --split-dir data/splits --config configs/default.yaml --out runs/xgb --hpo_trials 20 --horizon 3600
# afds train deepsurv --split-dir data/splits --config configs/default.yaml --out runs/deeps --hpo_trials 20 --horizon 3600

# Plots
afds plot --run runs/cox --what calibration,km --horizon 3600 --thresholds "0.02,0.07"
```

## Category lookup

| Column    | Code | Meaning                                  |
|-----------|------|-------------------------------------------|
| race      | 0    | White                                     |
| race      | 1    | Black                                     |
| race      | 2    | Other                                     |
| marital   | 0    | Single                                    |
| marital   | 1    | Married                                   |
| marital   | 2    | Divorced/widowed                          |
| marital   | 3    | Separated                                 |
| insurance | 0    | Public                                    |
| insurance | 1    | Private                                   |
| insurance | 2    | Unknown                                   |
| Age_1     | 1    | 1 if age ∈ [65,74], else 0               |
| Age_2     | 1    | 1 if age ∈ [75,84], else 0               |
| Age_3     | 1    | 1 if age ≥ 85, else 0                    |
| Age_*     | 0    | All zeros ⇒ age ≤ 64                     |



---

### Pretrained model artifacts

This repository includes **pretrained XGB‑Cox model artifacts** provided by the user under:

```
runs/xgb_imported/artifacts_minimal_xgb_cox/
├─ feature_manifest.json
├─ baseline_hazard.json
├─ app_metadata.json
└─ xgb_cox_model.json
```

These files are included **as-is** for reference and reproducibility. No source code was modified to integrate them.
