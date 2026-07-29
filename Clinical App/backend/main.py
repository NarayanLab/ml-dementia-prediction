import json
import numpy as np
import pandas as pd
import xgboost as xgb
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os

app = FastAPI(title="Dementia Risk Assessment API", version="4.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "https://ml-dementia-prediction-1.onrender.com"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "model")
MODEL_FILE = "xgb_cox_22feature.json"

booster = None
feature_names = None
feature_means = None
baseline = None
dca_band = None


class PatientData(BaseModel):
    af_age: int
    marital_status: str
    weight: float
    height: float
    bmi: float
    diabetes: bool
    hypertension: bool
    stroke_tia: bool
    depression: bool
    cognitive_deficit: bool
    osteoarthritis: bool
    parkinson: bool
    ppi: bool
    insurance: int
    rr_interval: float
    qrs_duration: float
    sodium_value: float
    potassium_value: float
    creatinine_value: float
    calcium_mg_dl: float
    # Availability flags (True = available, False = missing)
    # Only Calcium and HCT have missingness flags surfaced in the UI — those were
    # the only _missing flags selected as important predictors. Other labs always
    # have values from the form, so their _missing flags stay 0 (set from means dict).
    calcium_available: bool
    hct_available: bool


class RiskResponse(BaseModel):
    risk_percentage: float
    risk_category: str
    risk_color: str
    low_threshold: float
    high_threshold: float


def load_artifacts():
    global booster, feature_names, feature_means, baseline, dca_band

    # 22-feature model: BIC-selected features refit with the 77-feature model's
    # best hyperparameters (see model/training_meta.json, test C-index 0.822).
    # Every one of the 22 comes from the form -- nothing is imputed at request
    # time.
    booster = xgb.Booster()
    booster.load_model(os.path.join(MODEL_DIR, MODEL_FILE))

    # The booster is the authority on column order. Do NOT derive this from the
    # means dict or from feature_importance_ranking.json -- that file is an
    # importance ranking, not the column order, and using it would silently
    # scramble the inputs.
    feature_names = booster.feature_names

    # Population means: form defaults, plus the fill value for Calcium when the
    # clinician marks it unavailable.
    with open(os.path.join(MODEL_DIR, "feature_means.json")) as f:
        feature_means = json.load(f)

    # Baseline survival and risk thresholds. These are properties of this
    # specific fit -- they must not be mixed with another model's.
    with open(os.path.join(MODEL_DIR, "baseline.json")) as f:
        baseline = json.load(f)
    with open(os.path.join(MODEL_DIR, "dca_band.json")) as f:
        dca_band = json.load(f)

    missing = set(feature_names) - set(feature_means)
    if missing:
        raise RuntimeError(f"feature_means is missing model features: {sorted(missing)}")
    if len(feature_names) != 22:
        raise RuntimeError(f"expected a 22-feature model, got {len(feature_names)}")

    print(f"Loaded model: {len(feature_names)} features, S0(t*={baseline['t_star']})={baseline['S0_tstar']:.4f}")


def calculate_absolute_risk(margin: float) -> float:
    S0 = baseline["S0_tstar"]
    hr = np.exp(float(margin))
    survival = S0 ** hr
    return 1.0 - survival


@app.on_event("startup")
async def startup_event():
    load_artifacts()


@app.get("/")
async def root():
    return {"message": "Dementia Risk Assessment API", "status": "running", "version": "4.0.0"}


@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "model_loaded": booster is not None,
        "model_file": MODEL_FILE,
        "features_count": len(feature_names) if feature_names else 0,
        "t_star_years": baseline["t_star"] if baseline else None,
        "S0_tstar": baseline["S0_tstar"] if baseline else None,
        "dca_band": dca_band["band"] if dca_band else None,
    }


MARITAL_CODES = {"Single": 0, "Married": 1, "Divorced/Widowed": 2, "Unknown": 3}

# Features the form does not supply directly, so they may legitimately remain at
# their population mean. Calcium is filled from the mean when the clinician marks
# it unavailable; everything else in the 22 must come from the request.
MEAN_FILLED_OK = {"Calcium Value 3mo"}


def build_feature_row(patient: PatientData) -> dict:
    """Map a request onto the model's 22 features.

    Seeded from population means so a missing key can never become NaN, but every
    feature except Calcium is then overwritten from the form. Kept separate from
    the endpoint so tests can assert that directly.
    """
    row = dict(feature_means)

    row["AF_Age"] = float(patient.af_age)
    row["weight"] = patient.weight
    row["height"] = patient.height
    row["bmi"] = patient.bmi
    row["Marital"] = MARITAL_CODES[patient.marital_status]
    row["Insurance"] = patient.insurance
    row["DM"] = int(patient.diabetes)
    row["HTN"] = int(patient.hypertension)
    row["TIA_CVA_Stroke"] = int(patient.stroke_tia)
    row["Depression"] = int(patient.depression)
    row["Cognitive_Deficit"] = int(patient.cognitive_deficit)
    row["Osteoarthritis"] = int(patient.osteoarthritis)
    row["Parkinson"] = int(patient.parkinson)
    row["PPI"] = int(patient.ppi)

    # Labs always provided by the form. Their _missing flags were not selected
    # into the 22, so there is nothing to set alongside them.
    row["Sodium Value 3mo"] = patient.sodium_value
    row["Potassium Value 3mo"] = patient.potassium_value
    row["Creatinine Value 3mo"] = patient.creatinine_value
    row["RR Value 3mo"] = patient.rr_interval
    row["QRS Value 3mo"] = patient.qrs_duration  # QRS axis in degrees, can be negative

    # Calcium: availability surfaced in UI (Calcium_missing is one of the 22)
    if patient.calcium_available:
        row["Calcium Value 3mo"] = patient.calcium_mg_dl
        row["Calcium_missing"] = 0
    else:
        row["Calcium_missing"] = 1

    # HCT: flag-only feature, surfaced in UI (HCT_missing is one of the 22)
    row["HCT_missing"] = 0 if patient.hct_available else 1

    return row


@app.post("/predict", response_model=RiskResponse)
async def predict_risk(patient: PatientData):
    if booster is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    try:
        row = build_feature_row(patient)

        # Build DataFrame in the model's expected feature order
        X_row = pd.DataFrame([row], columns=feature_names)

        margin = booster.predict(xgb.DMatrix(X_row), output_margin=True)[0]
        risk = calculate_absolute_risk(margin)
        risk_percentage = risk * 100

        low_threshold, high_threshold = dca_band["band"]

        if risk <= low_threshold:
            risk_category = "Low Risk"
            risk_color = "#4CAF50"
        elif risk <= high_threshold:
            risk_category = "Medium Risk"
            risk_color = "#FFC107"
        else:
            risk_category = "High Risk"
            risk_color = "#F44336"

        return RiskResponse(
            risk_percentage=risk_percentage,
            risk_category=risk_category,
            risk_color=risk_color,
            low_threshold=low_threshold * 100,
            high_threshold=high_threshold * 100,
        )

    except KeyError as e:
        raise HTTPException(status_code=400, detail=f"Missing or invalid field: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
