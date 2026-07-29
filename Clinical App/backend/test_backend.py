"""
Verify the 22-feature XGBoost-Cox backend.

Guards the failure modes that actually bit this project:
  - a booster with the wrong feature count getting deployed
  - column order taken from the wrong file (feature_importance_ranking.json is an
    importance ranking, NOT the column order)
  - a feature silently left at its population mean instead of coming from the form

Run from the backend/ directory:  python test_backend.py
"""
import asyncio
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import main
from main import (MEAN_FILLED_OK, MODEL_DIR, MODEL_FILE, PatientData,
                  build_feature_row, load_artifacts, predict_risk)

failures = []


def check(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"   {detail}" if detail and not cond else ""))
    if not cond:
        failures.append(name)


def patient(**kw):
    """Reference patient at population means, overridable per test."""
    args = dict(
        af_age=70, marital_status="Married", weight=79.3, height=170.0, bmi=27.2,
        diabetes=False, hypertension=False, stroke_tia=False, depression=False,
        cognitive_deficit=False, osteoarthritis=False, parkinson=False, ppi=False,
        insurance=0, rr_interval=778, qrs_duration=17, sodium_value=138.5,
        potassium_value=4.2, creatinine_value=1.13, calcium_mg_dl=8.6,
        calcium_available=True, hct_available=True,
    )
    args.update(kw)
    return PatientData(**args)


def risk(**kw):
    return asyncio.run(predict_risk(patient(**kw))).risk_percentage


print("=" * 68)
print("22-feature dementia risk backend")
print("=" * 68)

print("\n1. Artifacts")
load_artifacts()
check("model loads", main.booster is not None)
check("exactly 22 features", len(main.feature_names) == 22, f"got {len(main.feature_names)}")
check("objective is survival:cox",
      json.loads(main.booster.save_config())["learner"]["objective"]["name"] == "survival:cox")

with open(os.path.join(MODEL_DIR, "meta.json")) as f:
    meta_names = json.load(f)["feature_names"]
check("column order matches meta.json", main.feature_names == meta_names)

with open(os.path.join(MODEL_DIR, "feature_importance_ranking.json")) as f:
    ranking = json.load(f)["order"]
check("column order is NOT feature_importance_ranking.json (that file is a ranking)",
      main.feature_names != ranking)

with open(os.path.join(MODEL_DIR, "training_meta.json")) as f:
    tmeta = json.load(f)
check("training_meta agrees with the booster's feature list",
      tmeta["feature_names"] == main.feature_names and tmeta["n_features"] == 22)
check("feature_means covers all 22", set(main.feature_names) <= set(main.feature_means))
print(f"        model_file={MODEL_FILE}  S0={main.baseline['S0_tstar']:.6f}  band={main.dca_band['band']}")

print("\n2. Every feature comes from the form, not from a mean")
# Deliberately off-mean values so any feature still sitting at its population
# mean is a wiring bug rather than a coincidence.
row = build_feature_row(patient(
    af_age=81, weight=91.5, height=161.0, bmi=35.3, diabetes=True, hypertension=True,
    stroke_tia=True, depression=True, cognitive_deficit=True, osteoarthritis=True,
    parkinson=True, ppi=True, insurance=2, rr_interval=911, qrs_duration=-44,
    sodium_value=133.0, potassium_value=5.1, creatinine_value=2.4, calcium_mg_dl=9.9,
    marital_status="Divorced/Widowed"))
check("row has exactly the 22 model features", set(row) == set(main.feature_names),
      f"extra={sorted(set(row) - set(main.feature_names))} missing={sorted(set(main.feature_names) - set(row))}")
stuck = [f for f in main.feature_names
         if f not in MEAN_FILLED_OK and abs(row[f] - main.feature_means[f]) < 1e-9]
check("no feature left at its population mean", not stuck, f"stuck at mean: {stuck}")

print("\n3. Risk is a valid probability")
for label, kw in [("reference", {}), ("youngest", dict(af_age=40)),
                  ("oldest + all comorbid", dict(af_age=95, diabetes=True, hypertension=True,
                                                 stroke_tia=True, depression=True,
                                                 cognitive_deficit=True, osteoarthritis=True,
                                                 parkinson=True, ppi=True))]:
    r = risk(**kw)
    check(f"0 < risk < 100  ({label}: {r:.2f}%)", 0.0 < r < 100.0)

print("\n4. Monotone in age")
ages = [55, 60, 65, 70, 75, 80, 85, 90]
risks = [risk(af_age=a) for a in ages]
print("        " + "  ".join(f"{a}:{r:.2f}%" for a, r in zip(ages, risks)))
check("non-decreasing across 55-90", all(b >= a - 1e-9 for a, b in zip(risks, risks[1:])))
check("age has an effect below 65 (the 20-feature model was flat there)",
      risk(af_age=55) != risk(af_age=64))

print("\n5. Known risk factors move risk upward")
base = risk(af_age=80)
for label, kw in [("cognitive deficit", dict(cognitive_deficit=True)),
                  ("Parkinson", dict(parkinson=True)),
                  ("stroke/TIA", dict(stroke_tia=True)),
                  ("depression", dict(depression=True)),
                  ("diabetes", dict(diabetes=True))]:
    r = risk(af_age=80, **kw)
    check(f"{label}: {base:.2f}% -> {r:.2f}%", r > base)

print("\n6. Availability toggles are wired to their _missing flags")
check("calcium unavailable changes the prediction",
      risk(calcium_available=True) != risk(calcium_available=False))
check("HCT unavailable changes the prediction",
      risk(hct_available=True) != risk(hct_available=False))

print("\n7. Marital encoding matches the data dictionary")
# 0 Single, 1 Married, 2 Divorced/Widowed, 3 Unknown
check("all four marital codes accepted and encoded per dictionary",
      [build_feature_row(patient(marital_status=s))["Marital"]
       for s in ("Single", "Married", "Divorced/Widowed", "Unknown")] == [0, 1, 2, 3])

print("\n8. Risk categories follow the DCA band")
low, high = main.dca_band["band"]
for kw in [{}, dict(af_age=80), dict(af_age=95, cognitive_deficit=True, parkinson=True)]:
    resp = asyncio.run(predict_risk(patient(**kw)))
    p = resp.risk_percentage / 100.0
    expected = "Low Risk" if p <= low else ("Medium Risk" if p <= high else "High Risk")
    check(f"{resp.risk_percentage:6.2f}% -> {resp.risk_category}", resp.risk_category == expected)

print("\n" + "=" * 68)
if failures:
    print(f"{len(failures)} CHECK(S) FAILED: {failures}")
    sys.exit(1)
print("All checks passed.")
print("=" * 68)
