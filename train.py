"""
usage:
    python train.py
    python train.py --experiment-name "experiment_v2" --n-estimators 200

MLflow UI:
    mlflow ui --backend-store-uri sqlite:///mlflow_tracking/mlflow.db
    open http://localhost:5000

"""

import argparse
import json
import os
import pickle
import warnings
from pathlib import Path

import mlflow
import mlflow.xgboost
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.impute import KNNImputer
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.preprocessing import RobustScaler

warnings.filterwarnings("ignore")

# ── Constants ─────────────────────────────────────────────────────────────────
DATA_PATH    = "./dataset/diabetes.csv"
ARTIFACT_DIR = Path("model_artifacts")
ZERO_COLS    = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
FEATURES     = [
    "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
    "Insulin", "BMI", "DiabetesPedigreeFunction", "Age",
]
RANDOM_STATE = 42


# ── CLI args ──────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="Train diabetes XGBoost classifier")
    p.add_argument("--experiment-name", default="diabetes_classification")
    p.add_argument("--n-estimators",    type=int,   default=100)
    p.add_argument("--max-depth",       type=int,   default=3)
    p.add_argument("--learning-rate",   type=float, default=0.05)
    p.add_argument("--subsample",       type=float, default=0.9)
    p.add_argument("--colsample",       type=float, default=0.9)
    p.add_argument("--test-size",       type=float, default=0.2)
    return p.parse_args()


# ── Preprocessing ─────────────────────────────────────────────────────────────
def load_and_preprocess(test_size: float):
    df = pd.read_csv(DATA_PATH)
    df[ZERO_COLS] = df[ZERO_COLS].replace(0, np.nan)

    X = df[FEATURES]
    y = df["Outcome"]

    # Fit imputer & scaler on FULL dataset (saved for inference)
    imputer = KNNImputer(n_neighbors=5)
    X_imp   = pd.DataFrame(imputer.fit_transform(X), columns=FEATURES)

    scaler  = RobustScaler()
    X_sc    = scaler.fit_transform(X_imp)

    X_train, X_test, y_train, y_test = train_test_split(
        X_sc, y, test_size=test_size, random_state=RANDOM_STATE, stratify=y
    )
    return X_train, X_test, y_train, y_test, imputer, scaler, df


# ── Training ──────────────────────────────────────────────────────────────────
def build_model(params: dict, scale_pos_weight: float) -> xgb.XGBClassifier:
    return xgb.XGBClassifier(
        n_estimators      = params["n_estimators"],
        max_depth         = params["max_depth"],
        learning_rate     = params["learning_rate"],
        subsample         = params["subsample"],
        colsample_bytree  = params["colsample_bytree"],
        scale_pos_weight  = scale_pos_weight,
        random_state      = RANDOM_STATE,
        eval_metric       = "logloss",
        tree_method       = "hist",
    )


def evaluate(model, X_test, y_test, threshold: float = 0.5):
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)

    metrics = {
        "roc_auc":         round(roc_auc_score(y_test, y_prob), 4),
        "avg_precision":   round(average_precision_score(y_test, y_prob), 4),
        "f1_macro":        round(f1_score(y_test, y_pred, average="macro"), 4),
        "f1_diabetic":     round(f1_score(y_test, y_pred, pos_label=1), 4),
        "accuracy":        round((y_pred == y_test.values).mean(), 4),
    }
    report = classification_report(
        y_test, y_pred, target_names=["No Diabetes", "Diabetes"]
    )
    cm = confusion_matrix(y_test, y_pred)
    return metrics, report, cm, y_prob


def cross_validate(model, X, y, cv=5):
    skf    = StratifiedKFold(n_splits=cv, shuffle=True, random_state=RANDOM_STATE)
    scores = cross_val_score(model, X, y, cv=skf, scoring="roc_auc", n_jobs=-1)
    return {"cv_mean_auc": round(scores.mean(), 4), "cv_std_auc": round(scores.std(), 4)}


# ── Save artifacts ────────────────────────────────────────────────────────────
def save_artifacts(model, imputer, scaler, metadata: dict):
    ARTIFACT_DIR.mkdir(exist_ok=True)

    # XGBoost model (native format — most portable)
    model_path = ARTIFACT_DIR / "xgb_model.json"
    model.save_model(str(model_path))

    # Preprocessors
    with open(ARTIFACT_DIR / "imputer.pkl", "wb") as f:
        pickle.dump(imputer, f)
    with open(ARTIFACT_DIR / "scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)

    # Metadata (features list, threshold, version)
    with open(ARTIFACT_DIR / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nArtifacts saved to  ./{ARTIFACT_DIR}/")
    for p in sorted(ARTIFACT_DIR.iterdir()):
        print(f"  {p.name}  ({p.stat().st_size:,} bytes)")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()

    # MLflow setup
    tracking_uri = f"sqlite:///mlflow_tracking/mlflow.db"
    os.makedirs("mlflow_tracking", exist_ok=True)
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(args.experiment_name)

    print(f"\n{'='*60}")
    print(f"  Diabetes Classifier Training Pipeline")
    print(f"  Experiment : {args.experiment_name}")
    print(f"{'='*60}\n")

    # Load data
    X_train, X_test, y_train, y_test, imputer, scaler, df = load_and_preprocess(
        args.test_size
    )
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    print(f"Train: {len(X_train)}  |  Test: {len(X_test)}")
    print(f"Class ratio (neg/pos): {scale_pos_weight:.2f}")

    params = {
        "n_estimators":    args.n_estimators,
        "max_depth":       args.max_depth,
        "learning_rate":   args.learning_rate,
        "subsample":       args.subsample,
        "colsample_bytree": args.colsample,
    }

    with mlflow.start_run() as run:
        print(f"\nMLflow run ID: {run.info.run_id}")

        # ── Log parameters ──────────────────────────────────────────────────
        mlflow.log_params(params)
        mlflow.log_param("test_size",       args.test_size)
        mlflow.log_param("random_state",    RANDOM_STATE)
        mlflow.log_param("imputer",         "KNNImputer(k=5)")
        mlflow.log_param("scaler",          "RobustScaler")
        mlflow.log_param("features",        str(FEATURES))
        mlflow.log_param("train_samples",   len(X_train))
        mlflow.log_param("test_samples",    len(X_test))
        mlflow.log_param("scale_pos_weight", round(scale_pos_weight, 3))

        # ── Train ────────────────────────────────────────────────────────────
        model = build_model(params, scale_pos_weight)
        model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            verbose=False,
        )
        print("Training complete.")

        # ── Cross-validation ─────────────────────────────────────────────────
        # Fit a fresh model on all data for CV (no test leakage in CV)
        cv_model  = build_model(params, scale_pos_weight)
        import numpy as _np
        X_all = _np.vstack([X_train, X_test])
        y_all = pd.concat([y_train, y_test]).reset_index(drop=True)
        cv_metrics = cross_validate(cv_model, X_all, y_all)
        mlflow.log_metrics(cv_metrics)
        print(f"  CV ROC-AUC: {cv_metrics['cv_mean_auc']} ± {cv_metrics['cv_std_auc']}")

        # ── Evaluate ──────────────────────────────────────────────────────────
        metrics, report, cm, y_prob = evaluate(model, X_test, y_test)
        mlflow.log_metrics(metrics)

        print(f"\n  ROC-AUC        : {metrics['roc_auc']}")
        print(f"  Avg Precision  : {metrics['avg_precision']}")
        print(f"  F1 (Diabetic)  : {metrics['f1_diabetic']}")
        print(f"  Accuracy       : {metrics['accuracy']}")
        print(f"\n{report}")

        # ── Log classification report as text artifact ────────────────────────
        report_path = ARTIFACT_DIR / "classification_report.txt"
        ARTIFACT_DIR.mkdir(exist_ok=True)
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(f"Run ID: {run.info.run_id}\n\n")
            f.write("Parameters:\n")
            for k, v in params.items():
                f.write(f"  {k}: {v}\n")
            f.write(f"\nMetrics:\n")
            for k, v in {**metrics, **cv_metrics}.items():
                f.write(f"  {k}: {v}\n")
            f.write(f"\nClassification Report:\n{report}")
            f.write(f"\nConfusion Matrix:\n{cm}")
        mlflow.log_artifact(str(report_path))

        # ── Log model with MLflow ─────────────────────────────────────────────
        mlflow.xgboost.log_model(
            model,
            artifact_path="xgb_model",
            input_example=pd.DataFrame([X_test[0]], columns=FEATURES),
        )

        # ── Set tags ──────────────────────────────────────────────────────────
        mlflow.set_tag("model_type",  "XGBoostClassifier")
        mlflow.set_tag("dataset",     "Pima Indians Diabetes")
        mlflow.set_tag("status",      "production" if metrics["roc_auc"] > 0.80 else "staging")
        mlflow.set_tag("best_feature","Glucose")

        # ── Save local artifacts ──────────────────────────────────────────────
        metadata = {
            "run_id":        run.info.run_id,
            "features":      FEATURES,
            "threshold":     0.5,
            "roc_auc":       metrics["roc_auc"],
            "model_version": "1.0",
            "experiment":    args.experiment_name,
        }
        save_artifacts(model, imputer, scaler, metadata)

        # Log preprocessor artifacts to MLflow too
        mlflow.log_artifact(str(ARTIFACT_DIR / "imputer.pkl"))
        mlflow.log_artifact(str(ARTIFACT_DIR / "scaler.pkl"))
        mlflow.log_artifact(str(ARTIFACT_DIR / "metadata.json"))
        mlflow.log_artifact(str(ARTIFACT_DIR / "xgb_model.json"))

        print(f"\nMLflow run complete.")
        print(f"  Tracking URI : {tracking_uri}")
        print(f"  Run ID       : {run.info.run_id}")
        print(f"\nView UI with:  mlflow ui --backend-store-uri sqlite:///mlflow_tracking/mlflow.db")


if __name__ == "__main__":
    main()
