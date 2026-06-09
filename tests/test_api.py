"""
tests/test_api.py
=================
Automated tests for the Diabetes Prediction API.

Covers:
  - Health endpoints
  - Valid single prediction
  - Batch prediction
  - Edge cases: zeros (missing values), boundary values
  - Input validation errors (out-of-range, missing fields)

Run:
    pytest tests/test_api.py -v
"""

import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

# Make sure app is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app.main import app

client = TestClient(app)

# ── Fixture ───────────────────────────────────────────────────────────────────
HEALTHY_PATIENT = {
    "Pregnancies": 1,
    "Glucose": 89,
    "BloodPressure": 66,
    "SkinThickness": 23,
    "Insulin": 94,
    "BMI": 28.1,
    "DiabetesPedigreeFunction": 0.167,
    "Age": 21,
}

HIGH_RISK_PATIENT = {
    "Pregnancies": 8,
    "Glucose": 183,
    "BloodPressure": 64,
    "SkinThickness": 0,   # missing → imputed
    "Insulin": 0,         # missing → imputed
    "BMI": 23.3,
    "DiabetesPedigreeFunction": 0.672,
    "Age": 32,
}


# ── Health tests ──────────────────────────────────────────────────────────────
class TestHealth:
    def test_root_returns_200(self):
        r = client.get("/")
        assert r.status_code == 200
        assert r.json()["status"] == "running"

    def test_health_returns_model_info(self):
        r = client.get("/health")
        assert r.status_code == 200
        data = r.json()
        assert data["status"] == "healthy"
        assert "roc_auc" in data
        assert "model_version" in data

    def test_features_endpoint(self):
        r = client.get("/features")
        assert r.status_code == 200
        features = r.json()["features"]
        assert "Glucose" in features
        assert "BMI" in features
        assert len(features) == 8

    def test_metrics_endpoint(self):
        r = client.get("/metrics")
        assert r.status_code == 200
        data = r.json()
        assert "roc_auc" in data


# ── Prediction tests ──────────────────────────────────────────────────────────
class TestPredict:
    def test_healthy_patient_prediction(self):
        r = client.post("/predict", json=HEALTHY_PATIENT)
        assert r.status_code == 200
        data = r.json()
        assert "probability" in data
        assert "prediction" in data
        assert "label" in data
        assert "risk_band" in data
        assert "latency_ms" in data
        assert 0.0 <= data["probability"] <= 1.0
        assert data["prediction"] in [0, 1]
        assert data["label"] in ["Diabetic", "Non-Diabetic"]
        assert data["risk_band"] in ["Low", "Medium", "High", "Critical"]

    def test_high_risk_patient_prediction(self):
        r = client.post("/predict", json=HIGH_RISK_PATIENT)
        assert r.status_code == 200
        data = r.json()
        # High-risk patient should have high probability
        assert data["probability"] > 0.5

    def test_zeros_are_handled_as_missing(self):
        """Zeros in Glucose, BP, etc. should not crash — they are imputed."""
        patient_with_zeros = {**HEALTHY_PATIENT, "Glucose": 0, "Insulin": 0, "BMI": 0}
        r = client.post("/predict", json=patient_with_zeros)
        assert r.status_code == 200

    def test_response_has_latency(self):
        r = client.post("/predict", json=HEALTHY_PATIENT)
        assert r.json()["latency_ms"] > 0

    def test_probability_is_float(self):
        r = client.post("/predict", json=HEALTHY_PATIENT)
        assert isinstance(r.json()["probability"], float)


# ── Batch tests ───────────────────────────────────────────────────────────────
class TestBatchPredict:
    def test_batch_with_two_patients(self):
        payload = {"patients": [HEALTHY_PATIENT, HIGH_RISK_PATIENT]}
        r = client.post("/predict/batch", json=payload)
        assert r.status_code == 200
        data = r.json()
        assert data["count"] == 2
        assert len(data["predictions"]) == 2

    def test_batch_returns_correct_count(self):
        payload = {"patients": [HEALTHY_PATIENT] * 10}
        r = client.post("/predict/batch", json=payload)
        assert r.json()["count"] == 10

    def test_batch_exceeds_limit(self):
        payload = {"patients": [HEALTHY_PATIENT] * 501}
        r = client.post("/predict/batch", json=payload)
        assert r.status_code == 400

    def test_single_patient_batch(self):
        payload = {"patients": [HEALTHY_PATIENT]}
        r = client.post("/predict/batch", json=payload)
        assert r.status_code == 200
        assert r.json()["count"] == 1


# ── Validation tests ──────────────────────────────────────────────────────────
class TestValidation:
    def test_missing_field_returns_422(self):
        incomplete = {k: v for k, v in HEALTHY_PATIENT.items() if k != "Glucose"}
        r = client.post("/predict", json=incomplete)
        assert r.status_code == 422

    def test_glucose_out_of_range_returns_422(self):
        bad = {**HEALTHY_PATIENT, "Glucose": 999}
        r = client.post("/predict", json=bad)
        assert r.status_code == 422

    def test_bmi_out_of_range_returns_422(self):
        bad = {**HEALTHY_PATIENT, "BMI": -5}
        r = client.post("/predict", json=bad)
        assert r.status_code == 422

    def test_age_out_of_range_returns_422(self):
        bad = {**HEALTHY_PATIENT, "Age": 200}
        r = client.post("/predict", json=bad)
        assert r.status_code == 422

    def test_empty_body_returns_422(self):
        r = client.post("/predict", json={})
        assert r.status_code == 422
