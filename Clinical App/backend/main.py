import json
import numpy as np
import pandas as pd
import xgboost as xgb
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import os

app = FastAPI(title="Dementia Risk Assessment API", version="2.0.0")

# Enable CORS for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # React dev server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for model artifacts
booster = None
feature_manifest = None
app_metadata = None
baseline_hazard = None

class PatientData(BaseModel):
    af_age: int
    marital_status: str  # "Single", "Married", "Divorced/Widowed"
    weight: float
    bmi: float
    diabetes: bool
    hypertension: bool
    stroke_tia: bool
    depression: bool
    cognitive_impairment: bool
    rr_interval: float
    qrs_duration: int
    sodium_value: float
    calcium_mg_dl: float
    osteoarthritis: bool
    race: int  # Encoded race value
    insurance: int  # Encoded insurance value
    osteoporosis: bool
    parkinson: bool

class RiskResponse(BaseModel):
    risk_percentage: float
    risk_category: str
    risk_color: str
    low_threshold: float
    high_threshold: float

def load_artifacts():
    """Load all model artifacts and configuration files"""
    global booster, feature_manifest, app_metadata, baseline_hazard

    try:
        # Load XGBoost model
        booster = xgb.Booster()
        booster.load_model("../xgb_cox_model.json")

        # Load new configuration files
        with open("../feature_manifest.json") as f:
            feature_manifest = json.load(f)
        with open("../app_metadata.json") as f:
            app_metadata = json.load(f)
        with open("../baseline_hazard.json") as f:
            baseline_hazard = json.load(f)

        print("Model artifacts loaded successfully")
        print(f"Features: {len(feature_manifest['feature_names'])}")
        print(f"Horizon: {app_metadata['horizon_days']} days")

    except Exception as e:
        print(f"Error loading model artifacts: {e}")
        raise e

def predict_margin(booster, Xrow_df):
    """Get model prediction margin (log hazard ratio)"""
    return booster.predict(xgb.DMatrix(Xrow_df), output_margin=True)

def calculate_absolute_risk(margin, event_times, cum_baseline_hazard, horizon_days):
    """
    Calculate absolute risk at specified horizon using cumulative baseline hazard.

    Args:
        margin: XGBoost prediction margin (log hazard ratio)
        event_times: Array of event times in years
        cum_baseline_hazard: Cumulative baseline hazard at each event time
        horizon_days: Prediction horizon in days (e.g., 1826 for 5 years)

    Returns:
        Absolute risk probability at horizon
    """
    # Convert horizon from days to years
    horizon_years = horizon_days / 365.25

    # Find the cumulative baseline hazard at the horizon
    event_times_array = np.array(event_times)
    cum_hazard_array = np.array(cum_baseline_hazard)

    # Find the index where event_time >= horizon_years
    idx = np.searchsorted(event_times_array, horizon_years, side='right') - 1
    idx = max(0, min(idx, len(cum_hazard_array) - 1))

    H0_t = cum_hazard_array[idx]

    # Calculate hazard ratio from margin
    hazard_ratio = np.exp(float(margin[0]))

    # Calculate survival probability: S(t) = exp(-H0(t) * exp(margin))
    survival_prob = np.exp(-H0_t * hazard_ratio)

    # Risk is 1 - survival
    risk = 1.0 - survival_prob

    return risk

@app.on_event("startup")
async def startup_event():
    """Load model artifacts on startup"""
    load_artifacts()

@app.get("/")
async def root():
    return {"message": "Dementia Risk Assessment API", "status": "running"}

@app.post("/predict", response_model=RiskResponse)
async def predict_risk(patient: PatientData):
    """Calculate dementia risk for a patient using the 20-feature model"""

    try:
        if booster is None:
            raise HTTPException(status_code=500, detail="Model not loaded")

        # Get model configuration
        feature_names = feature_manifest["feature_names"]
        risk_thresholds = app_metadata["risk_thresholds"]
        horizon_days = app_metadata["horizon_days"]
        event_times = baseline_hazard["event_times"]
        cum_baseline_hazard = baseline_hazard["cum_baseline_hazard"]

        # Convert AF age to categorical variables
        af_age_2 = 1 if 65 <= patient.af_age < 75 else 0
        af_age_3 = 1 if 75 <= patient.af_age < 85 else 0
        af_age_4 = 1 if patient.af_age >= 85 else 0

        # Convert marital status to numeric (0=Single, 1=Married, 2=Divorced/Widowed, 3=Unknown)
        marital_code = {"Single": 0, "Married": 1, "Divorced/Widowed": 2, "Unknown": 3}[patient.marital_status]

        # Build feature dictionary with ONLY the 20 features the model expects
        row = {
            "AF_age_4": af_age_4,
            "AF_age_3": af_age_3,
            "weight": patient.weight,
            "AF_age_2": af_age_2,
            "bmi": patient.bmi,
            "DM": 1 if patient.diabetes else 0,
            "Marital": marital_code,
            "TIA_CVA_Stroke": 1 if patient.stroke_tia else 0,
            "RR Value 3mo": patient.rr_interval,
            "Depression": 1 if patient.depression else 0,
            "QRS Value 3mo": patient.qrs_duration,
            "HTN": 1 if patient.hypertension else 0,
            "Cognitive_Impairment": 1 if patient.cognitive_impairment else 0,
            "Sodium Value 3mo": patient.sodium_value,
            "Calcium Value 3mo": patient.calcium_mg_dl,
            "Osteoarthritis": 1 if patient.osteoarthritis else 0,
            "race": patient.race,
            "Insurance": patient.insurance,
            "Osteoporosis": 1 if patient.osteoporosis else 0,
            "Parkinson": 1 if patient.parkinson else 0,
        }

        # Create DataFrame with features in the exact order expected by the model
        X_row = pd.DataFrame([row], columns=feature_names)

        # Calculate risk using the XGBoost model
        margin = predict_margin(booster, X_row)
        risk = calculate_absolute_risk(
            margin,
            event_times,
            cum_baseline_hazard,
            horizon_days
        )
        risk_percentage = risk * 100

        # Determine risk category and color
        low_threshold = risk_thresholds["low"]
        high_threshold = risk_thresholds["high"]

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
            high_threshold=high_threshold * 100
        )

    except KeyError as e:
        raise HTTPException(status_code=400, detail=f"Missing required field: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": booster is not None,
        "features_count": len(feature_manifest["feature_names"]) if feature_manifest else 0,
        "horizon_days": app_metadata["horizon_days"] if app_metadata else 0
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)