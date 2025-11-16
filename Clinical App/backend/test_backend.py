"""
Test script to verify backend functionality with the new 20-feature model
"""
import json
import sys
import os

# Set UTF-8 encoding for Windows console
if sys.platform == 'win32':
    os.system('chcp 65001 > nul 2>&1')

sys.path.insert(0, '..')

import pandas as pd
import xgboost as xgb
import numpy as np

print("=" * 60)
print("Testing Backend with 20-Feature Model")
print("=" * 60)

# Load artifacts
print("\n1. Loading model artifacts...")
try:
    booster = xgb.Booster()
    booster.load_model("../xgb_cox_model.json")
    print("   ✓ XGBoost model loaded")

    with open("../feature_manifest.json") as f:
        feature_manifest = json.load(f)
    print(f"   ✓ Feature manifest loaded: {len(feature_manifest['feature_names'])} features")

    with open("../app_metadata.json") as f:
        app_metadata = json.load(f)
    print(f"   ✓ App metadata loaded: {app_metadata['horizon_days']} days horizon")

    with open("../baseline_hazard.json") as f:
        baseline_hazard = json.load(f)
    print(f"   ✓ Baseline hazard loaded: {len(baseline_hazard['event_times'])} time points")

except Exception as e:
    print(f"   ✗ Error loading artifacts: {e}")
    sys.exit(1)

# Test patient data
print("\n2. Creating test patient data...")
test_patient = {
    "AF_age_4": 0,  # Age 70 -> not >= 85
    "AF_age_3": 0,  # Age 70 -> not 75-84
    "weight": 79.3,
    "AF_age_2": 1,  # Age 70 -> 65-74
    "bmi": 27.2,
    "DM": 0,
    "Marital": 2,  # Married
    "TIA_CVA_Stroke": 0,
    "RR Value 3mo": 778,
    "Depression": 0,
    "QRS Value 3mo": 100,
    "HTN": 0,
    "Cognitive_Impairment": 0,
    "Sodium Value 3mo": 138.5,
    "Calcium Value 3mo": 9.6,
    "Osteoarthritis": 0,
    "race": 0,  # White
    "Insurance": 0,  # Medicare
    "Osteoporosis": 0,
    "Parkinson": 0,
}
print(f"   ✓ Test patient created with {len(test_patient)} features")

# Verify feature order
feature_names = feature_manifest["feature_names"]
print("\n3. Verifying feature order...")
for i, feat_name in enumerate(feature_names):
    if feat_name not in test_patient:
        print(f"   ✗ Missing feature: {feat_name}")
        sys.exit(1)
print(f"   ✓ All {len(feature_names)} features present")

# Create DataFrame
print("\n4. Creating feature DataFrame...")
X_row = pd.DataFrame([test_patient], columns=feature_names)
print(f"   ✓ DataFrame shape: {X_row.shape}")
print(f"   ✓ Columns: {list(X_row.columns)}")

# Make prediction
print("\n5. Making prediction...")
try:
    margin = booster.predict(xgb.DMatrix(X_row), output_margin=True)
    print(f"   ✓ Margin (log hazard ratio): {margin[0]:.6f}")

    # Calculate risk
    horizon_years = app_metadata["horizon_days"] / 365.25
    event_times_array = np.array(baseline_hazard["event_times"])
    cum_hazard_array = np.array(baseline_hazard["cum_baseline_hazard"])

    idx = np.searchsorted(event_times_array, horizon_years, side='right') - 1
    idx = max(0, min(idx, len(cum_hazard_array) - 1))
    H0_t = cum_hazard_array[idx]

    hazard_ratio = np.exp(float(margin[0]))
    survival_prob = np.exp(-H0_t * hazard_ratio)
    risk = 1.0 - survival_prob
    risk_percentage = risk * 100

    print(f"   ✓ Cumulative baseline hazard at {horizon_years:.2f} years: {H0_t:.6f}")
    print(f"   ✓ Hazard ratio: {hazard_ratio:.6f}")
    print(f"   ✓ Survival probability: {survival_prob:.6f}")
    print(f"   ✓ Risk: {risk_percentage:.2f}%")

    # Determine risk category
    low_threshold = app_metadata["risk_thresholds"]["low"]
    high_threshold = app_metadata["risk_thresholds"]["high"]

    if risk <= low_threshold:
        category = "Low Risk"
    elif risk <= high_threshold:
        category = "Medium Risk"
    else:
        category = "High Risk"

    print(f"   ✓ Risk category: {category}")

except Exception as e:
    print(f"   ✗ Prediction error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 60)
print("✓ All tests passed!")
print("=" * 60)
