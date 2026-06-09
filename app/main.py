"""
app/main.py
===========
FastAPI application for Diabetes Risk Prediction.

Endpoints:
  GET  /           → health check
  GET  /health     → detailed health + model metadata
  POST /predict    → single patient prediction
  POST /predict/batch → batch predictions (list of patients)
  GET  /features   → feature descriptions & valid ranges
  GET  /metrics    → latest model metrics (from metadata)
"""

import json
import pickle
import time
from functools import lru_cache
from pathlib import Path
from typing import Optional

import numpy as np
import xgboost as xgb
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator, model_validator

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR     = Path(__file__).resolve().parent.parent
ARTIFACT_DIR = BASE_DIR / "model_artifacts"

# ── Feature metadata ──────────────────────────────────────────────────────────
FEATURES = [
    "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
    "Insulin", "BMI", "DiabetesPedigreeFunction", "Age",
]

FEATURE_INFO = {
    "Pregnancies":             {"unit": "count",    "min": 0,    "max": 20,   "description": "Number of pregnancies"},
    "Glucose":                 {"unit": "mg/dL",    "min": 44,   "max": 199,  "description": "Plasma glucose (2-hour OGTT)"},
    "BloodPressure":           {"unit": "mmHg",     "min": 24,   "max": 122,  "description": "Diastolic blood pressure"},
    "SkinThickness":           {"unit": "mm",       "min": 7,    "max": 99,   "description": "Triceps skinfold thickness"},
    "Insulin":                 {"unit": "µU/mL",    "min": 14,   "max": 846,  "description": "2-hour serum insulin"},
    "BMI":                     {"unit": "kg/m²",    "min": 18.2, "max": 67.1, "description": "Body mass index"},
    "DiabetesPedigreeFunction":{"unit": "score",    "min": 0.07, "max": 2.42, "description": "Genetic diabetes risk function"},
    "Age":                     {"unit": "years",    "min": 21,   "max": 81,   "description": "Age in years"},
}

RISK_BANDS = [
    (0.00, 0.30, "Low",      "Routine lifestyle monitoring recommended."),
    (0.30, 0.55, "Medium",   "Diet & exercise programme advised. Follow-up in 6 months."),
    (0.55, 0.75, "High",     "Specialist referral + possible medication. Follow-up in 3 months."),
    (0.75, 1.01, "Critical", "Immediate clinical intervention required. Weekly follow-up."),
]


def get_risk_band(prob: float) -> dict:
    for lo, hi, label, advice in RISK_BANDS:
        if lo <= prob < hi:
            return {"band": label, "advice": advice}
    return {"band": "Critical", "advice": RISK_BANDS[-1][3]}


# ── Model loader ──────────────────────────────────────────────────────────────
class ModelStore:
    """Lazy-loaded, cached model store."""

    def __init__(self):
        self._model    = None
        self._imputer  = None
        self._scaler   = None
        self._metadata = None

    def _load(self):
        if self._model is not None:
            return
        try:
            self._model = xgb.XGBClassifier()
            self._model.load_model(str(ARTIFACT_DIR / "xgb_model.json"))

            with open(ARTIFACT_DIR / "imputer.pkl", "rb") as f:
                self._imputer = pickle.load(f)

            with open(ARTIFACT_DIR / "scaler.pkl", "rb") as f:
                self._scaler = pickle.load(f)

            with open(ARTIFACT_DIR / "metadata.json", "r", encoding="utf-8") as f:
                self._metadata = json.load(f)

        except FileNotFoundError as e:
            raise RuntimeError(
                f"Model artifacts not found. Run train.py first.\n{e}"
            )

    @property
    def model(self):
        self._load(); return self._model

    @property
    def imputer(self):
        self._load(); return self._imputer

    @property
    def scaler(self):
        self._load(); return self._scaler

    @property
    def metadata(self):
        self._load(); return self._metadata

    def predict(self, patient_values: list) -> dict:
        """Run full inference pipeline on a single patient row."""
        import pandas as pd

        # Build DataFrame to preserve feature names (avoids sklearn warning)
        X_raw = pd.DataFrame([patient_values], columns=FEATURES)

        # Impute → scale
        X_imp = pd.DataFrame(
            self.imputer.transform(X_raw), columns=FEATURES
        )
        X_sc  = self.scaler.transform(X_imp)

        prob     = float(self.model.predict_proba(X_sc)[0, 1])
        label    = int(prob >= self.metadata.get("threshold", 0.5))
        risk     = get_risk_band(prob)

        return {
            "probability":   round(prob, 4),
            "prediction":    label,
            "label":         "Diabetic" if label else "Non-Diabetic",
            "risk_band":     risk["band"],
            "advice":        risk["advice"],
        }


store = ModelStore()

# ── FastAPI app ───────────────────────────────────────────────────────────────
app = FastAPI(
    title       = "Diabetes Risk Prediction API",
    description = "XGBoost-based binary classifier for diabetes risk assessment.",
    version     = "1.0.0",
    docs_url    = "/docs",
    redoc_url   = "/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Pydantic schemas ──────────────────────────────────────────────────────────
class PatientInput(BaseModel):
    Pregnancies:              float = Field(..., ge=0,   le=20,  description="Number of pregnancies")
    Glucose:                  float = Field(..., ge=0,   le=250, description="Plasma glucose mg/dL (0 = unknown)")
    BloodPressure:            float = Field(..., ge=0,   le=140, description="Diastolic BP mmHg (0 = unknown)")
    SkinThickness:            float = Field(..., ge=0,   le=110, description="Triceps skinfold mm (0 = unknown)")
    Insulin:                  float = Field(..., ge=0,   le=900, description="2-h serum insulin µU/mL (0 = unknown)")
    BMI:                      float = Field(..., ge=0,   le=70,  description="Body mass index kg/m² (0 = unknown)")
    DiabetesPedigreeFunction: float = Field(..., ge=0.0, le=3.0, description="Genetic risk function score")
    Age:                      int   = Field(..., ge=1,   le=120, description="Age in years")

    model_config = {"json_schema_extra": {"example": {
        "Pregnancies": 2, "Glucose": 120, "BloodPressure": 72,
        "SkinThickness": 20, "Insulin": 80, "BMI": 28.5,
        "DiabetesPedigreeFunction": 0.35, "Age": 33,
    }}}

    def to_feature_list(self) -> list:
        return [
            self.Pregnancies, self.Glucose, self.BloodPressure,
            self.SkinThickness, self.Insulin, self.BMI,
            self.DiabetesPedigreeFunction, self.Age,
        ]


class PredictionResponse(BaseModel):
    probability: float
    prediction:  int
    label:       str
    risk_band:   str
    advice:      str
    latency_ms:  float


class BatchRequest(BaseModel):
    patients: list[PatientInput]


class BatchResponse(BaseModel):
    count:       int
    predictions: list[PredictionResponse]


# ── Endpoints ─────────────────────────────────────────────────────────────────
@app.get("/", tags=["Health"])
def root():
    return {
        "service":     "Diabetes Risk Prediction API",
        "version":     "1.0.0",
        "status":      "running",
        "docs":        "/docs",
    }


@app.get("/health", tags=["Health"])
def health():
    try:
        meta = store.metadata
        return {
            "status":          "healthy",
            "model_version":   meta.get("model_version", "unknown"),
            "roc_auc":         meta.get("roc_auc"),
            "run_id":          meta.get("run_id"),
            "features":        meta.get("features"),
            "threshold":       meta.get("threshold"),
        }
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Model not loaded: {e}")


@app.get("/features", tags=["Info"])
def feature_descriptions():
    return {"features": FEATURE_INFO}


@app.get("/metrics", tags=["Info"])
def model_metrics():
    try:
        return store.metadata
    except Exception as e:
        raise HTTPException(status_code=503, detail=str(e))


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
def predict(patient: PatientInput):
    """
    Predict diabetes risk for a single patient.

    - Returns probability, binary prediction, risk band, and clinical advice.
    - Values of 0 for Glucose, BloodPressure, SkinThickness, Insulin, BMI
      are treated as missing and imputed automatically.
    """
    t0 = time.perf_counter()
    try:
        result = store.predict(patient.to_feature_list())
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    latency = round((time.perf_counter() - t0) * 1000, 2)
    return PredictionResponse(**result, latency_ms=latency)


@app.post("/predict/batch", response_model=BatchResponse, tags=["Prediction"])
def predict_batch(batch: BatchRequest):
    """
    Predict diabetes risk for a list of patients (max 500).
    """
    if len(batch.patients) > 500:
        raise HTTPException(status_code=400, detail="Max 500 patients per batch request.")

    predictions = []
    for patient in batch.patients:
        t0 = time.perf_counter()
        result = store.predict(patient.to_feature_list())
        latency = round((time.perf_counter() - t0) * 1000, 2)
        predictions.append(PredictionResponse(**result, latency_ms=latency))

    return BatchResponse(count=len(predictions), predictions=predictions)
