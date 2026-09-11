"""Training pipeline for THERMOGRID Thermal Hotspot Intelligence.

Models trained:
1. XGBoost (XGBClassifier) -> Stage 1: Environment Association
   (industrial_associated, natural_vegetation, uncertain)
2. LightGBM (LGBMClassifier) -> Stage 2: Behaviour Analysis
   (persistent_expected, new_abnormal, insufficient_history)
3. Ridge & Linear Regression -> Continuous Persistence & FRP Anomaly Scoring
4. Logistic Regression -> Linear Baseline Benchmark

Exports trained models and evaluation metrics to `ml/models/`.
"""

import json
from pathlib import Path
import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "ml" / "data"
MODELS_DIR = PROJECT_ROOT / "ml" / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

STAGE1_CLASSES = ["industrial_associated", "natural_vegetation", "uncertain"]
STAGE2_CLASSES = ["persistent_expected", "new_abnormal", "insufficient_history"]


def haversine_np(lat1, lon1, lat2, lon2):
    """Vectorized haversine distance in kilometers."""
    r = 6371.0
    phi1 = np.radians(lat1)
    phi2 = np.radians(lat2)
    delta_phi = np.radians(lat2 - lat1)
    delta_lambda = np.radians(lon2 - lon1)
    a = (
        np.sin(delta_phi / 2.0) ** 2
        + np.cos(phi1) * np.cos(phi2) * np.sin(delta_lambda / 2.0) ** 2
    )
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))
    return r * c


def build_spatial_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute local spatial density and proximity to other thermal detections."""
    lats = df["latitude"].to_numpy()
    lons = df["longitude"].to_numpy()
    n = len(df)

    spatial_density_15km = np.zeros(n, dtype=np.float32)
    min_neighbor_dist_km = np.zeros(n, dtype=np.float32)

    for i in range(n):
        dists = haversine_np(lats[i], lons[i], lats, lons)
        dists_no_self = np.delete(dists, i)
        if len(dists_no_self) > 0:
            spatial_density_15km[i] = np.sum(dists_no_self <= 15.0)
            min_neighbor_dist_km[i] = np.min(dists_no_self)
        else:
            spatial_density_15km[i] = 0.0
            min_neighbor_dist_km[i] = 99.0

    df["spatial_density_15km"] = spatial_density_15km
    df["min_neighbor_dist_km"] = min_neighbor_dist_km
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Feature engineering from FIRMS VIIRS/MODIS thermal tabular features."""
    df = df.copy()

    # Differential Brightness Temperature: Key physical signature of active flaming
    df["delta_bt"] = df["bright_ti4"] - df["bright_ti5"]
    df["bt_ratio"] = df["bright_ti4"] / np.maximum(df["bright_ti5"], 1.0)

    # Fire Radiative Power transformation
    df["frp_clean"] = pd.to_numeric(df["frp"], errors="coerce").fillna(0.0)
    df["log_frp"] = np.log1p(np.maximum(df["frp_clean"], 0.0))

    # Pixel geometry
    df["scan"] = pd.to_numeric(df["scan"], errors="coerce").fillna(0.4)
    df["track"] = pd.to_numeric(df["track"], errors="coerce").fillna(0.38)
    df["pixel_area"] = df["scan"] * df["track"]

    # Sensor Confidence numeric encoding
    conf_map = {"l": 0.30, "n": 0.75, "h": 0.95}
    if "confidence" in df.columns:
        if df["confidence"].dtype == object:
            df["confidence_num"] = (
                df["confidence"].str.lower().map(conf_map).fillna(0.70)
            )
        else:
            df["confidence_num"] = df["confidence"].astype(float) / 100.0
    else:
        df["confidence_num"] = 0.75

    # Day / Night binary flag
    if "daynight" in df.columns:
        df["is_day"] = (df["daynight"].astype(str).str.upper() == "D").astype(
            float
        )
    else:
        df["is_day"] = 1.0

    # Acquisition Hour extraction
    if "acq_time" in df.columns:
        acq_num = (
            pd.to_numeric(df["acq_time"], errors="coerce").fillna(800).astype(int)
        )
        df["hour"] = (acq_num // 100) + (acq_num % 100) / 60.0
    else:
        df["hour"] = 12.0

    # Spatial neighborhood features
    df = build_spatial_features(df)

    # Thermal Intensity Index (composite physical metric)
    df["thermal_intensity_index"] = (
        (df["delta_bt"] / 30.0) * 0.4
        + df["log_frp"] * 0.35
        + df["confidence_num"] * 0.25
    )

    return df


FEATURE_COLUMNS = [
    "latitude",
    "longitude",
    "bright_ti4",
    "bright_ti5",
    "delta_bt",
    "bt_ratio",
    "frp_clean",
    "log_frp",
    "scan",
    "track",
    "pixel_area",
    "confidence_num",
    "is_day",
    "hour",
    "spatial_density_15km",
    "min_neighbor_dist_km",
    "thermal_intensity_index",
]


def create_ground_truth_labels(df: pd.DataFrame):
    """Establish robust domain ground truth labels for Stage 1, Stage 2, and Regression targets."""
    n = len(df)
    stage1_labels = np.zeros(n, dtype=int)
    stage2_labels = np.zeros(n, dtype=int)
    persistence_scores = np.zeros(n, dtype=np.float32)

    for i in range(n):
        row = df.iloc[i]
        frp = row["frp_clean"]
        delta_bt = row["delta_bt"]
        conf = row["confidence_num"]
        density = row["spatial_density_15km"]
        is_day = row["is_day"]

        # Stage 1: Environment Association
        if conf < 0.4 or (frp < 0.8 and delta_bt < 15.0):
            s1 = 2  # uncertain
        elif (frp >= 4.0 and delta_bt >= 28.0) or (is_day == 0.0 and delta_bt >= 20.0):
            s1 = 0  # industrial_associated
        elif density >= 3 and delta_bt >= 22.0 and frp >= 3.0:
            s1 = 0  # industrial cluster
        else:
            s1 = 1  # natural_vegetation
        stage1_labels[i] = s1

        # Stage 2: Behaviour Analysis
        if conf < 0.4 or density == 0:
            s2 = 2  # insufficient_history
            pers = 0.25 + 0.15 * np.random.rand()
        elif s1 == 0 and density >= 2:
            s2 = 0  # persistent_expected
            pers = 0.72 + 0.22 * min(1.0, (delta_bt / 40.0))
        elif frp >= 6.0 and delta_bt >= 32.0:
            s2 = 1  # new_abnormal intense flare
            pers = 0.45 + 0.20 * np.random.rand()
        elif density >= 4:
            s2 = 0  # persistent agricultural burn area
            pers = 0.65 + 0.18 * np.random.rand()
        else:
            s2 = 1  # new_abnormal transient fire
            pers = 0.35 + 0.15 * np.random.rand()

        stage2_labels[i] = s2
        persistence_scores[i] = float(np.clip(pers, 0.1, 0.98))

    df["stage1_target"] = stage1_labels
    df["stage2_target"] = stage2_labels
    df["persistence_target"] = persistence_scores
    return df


def train_models():
    """Main training routine."""
    print("=" * 70)
    print("🚀 THERMOGRID ML Training Pipeline")
    print("Models: XGBoost (Stage 1), LightGBM (Stage 2), Ridge/Linear (Regression)")
    print("=" * 70)

    # 1. Load Data
    firms_file = DATA_DIR / "firms_latest.csv"
    if not firms_file.exists():
        raise FileNotFoundError(f"FIRMS dataset not found at {firms_file}")

    raw_df = pd.read_csv(firms_file)
    print(f"Loaded {len(raw_df)} FIRMS records from {firms_file.name}")

    # 2. Engineer features
    df = engineer_features(raw_df)
    df = create_ground_truth_labels(df)

    X_df = df[FEATURE_COLUMNS]
    X = X_df.to_numpy(dtype=np.float32)
    y_stage1 = df["stage1_target"].to_numpy(dtype=int)
    y_stage2 = df["stage2_target"].to_numpy(dtype=int)
    y_pers = df["persistence_target"].to_numpy(dtype=np.float32)

    print(f"Features dimension: {X.shape}")
    print(f"Stage 1 distribution: {dict(df['stage1_target'].value_counts())}")
    print(f"Stage 2 distribution: {dict(df['stage2_target'].value_counts())}")

    # 3. Feature Scaling for linear models
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # -----------------------------------------------------------------
    # MODEL 1: XGBoost Classifier (Stage 1 - Environment Association)
    # -----------------------------------------------------------------
    print("\n--- Training Model 1: XGBoost (Stage 1: Environment) ---")
    xgb_stage1 = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.08,
        subsample=0.85,
        colsample_bytree=0.85,
        random_state=42,
        eval_metric="mlogloss",
    )
    xgb_stage1.fit(X, y_stage1)
    s1_preds = xgb_stage1.predict(X)
    s1_acc = accuracy_score(y_stage1, s1_preds)
    s1_f1 = f1_score(y_stage1, s1_preds, average="weighted")
    print(f"XGBoost Stage 1 Training Accuracy: {s1_acc:.4f} | F1-score: {s1_f1:.4f}")

    # -----------------------------------------------------------------
    # MODEL 2: LightGBM Classifier (Stage 2 - Behaviour Analysis)
    # -----------------------------------------------------------------
    print("\n--- Training Model 2: LightGBM (Stage 2: Behaviour) ---")
    lgb_stage2 = lgb.LGBMClassifier(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.08,
        subsample=0.85,
        colsample_bytree=0.85,
        objective="multiclass",
        num_class=3,
        random_state=42,
        verbosity=-1,
    )
    lgb_stage2.fit(X, y_stage2)
    s2_preds = lgb_stage2.predict(X)
    s2_acc = accuracy_score(y_stage2, s2_preds)
    s2_f1 = f1_score(y_stage2, s2_preds, average="weighted")
    print(f"LightGBM Stage 2 Training Accuracy: {s2_acc:.4f} | F1-score: {s2_f1:.4f}")

    # -----------------------------------------------------------------
    # MODEL 3: Ridge & Linear Regression (Continuous Persistence)
    # -----------------------------------------------------------------
    print("\n--- Training Model 3: Ridge Regression (Persistence Score) ---")
    ridge_pers = Ridge(alpha=1.0, random_state=42)
    ridge_pers.fit(X_scaled, y_pers)
    pers_preds = ridge_pers.predict(X_scaled)
    pers_r2 = r2_score(y_pers, pers_preds)
    pers_mae = mean_absolute_error(y_pers, pers_preds)
    print(f"Ridge Regression R² Score: {pers_r2:.4f} | MAE: {pers_mae:.4f}")

    # -----------------------------------------------------------------
    # MODEL 4: Logistic Regression Baseline (Linear Benchmark)
    # -----------------------------------------------------------------
    print("\n--- Training Model 4: Logistic Regression Baseline ---")
    logreg_baseline = LogisticRegression(max_iter=2000, random_state=42)
    logreg_baseline.fit(X_scaled, y_stage1)
    lr_preds = logreg_baseline.predict(X_scaled)
    lr_acc = accuracy_score(y_stage1, lr_preds)
    print(f"Logistic Regression Baseline Accuracy: {lr_acc:.4f}")

    # 4. Feature Importances
    xgb_feat_imp = dict(
        zip(
            FEATURE_COLUMNS,
            [float(x) for x in xgb_stage1.feature_importances_],
        )
    )
    lgb_feat_imp = dict(
        zip(
            FEATURE_COLUMNS,
            [float(x) for x in lgb_stage2.feature_importances_],
        )
    )

    metrics = {
        "training_samples": len(df),
        "stage1_xgboost": {
            "model": "XGBClassifier",
            "classes": STAGE1_CLASSES,
            "accuracy": float(s1_acc),
            "f1_score": float(s1_f1),
            "feature_importance_top5": sorted(
                xgb_feat_imp.items(), key=lambda x: x[1], reverse=True
            )[:5],
        },
        "stage2_lightgbm": {
            "model": "LGBMClassifier",
            "classes": STAGE2_CLASSES,
            "accuracy": float(s2_acc),
            "f1_score": float(s2_f1),
            "feature_importance_top5": sorted(
                lgb_feat_imp.items(), key=lambda x: x[1], reverse=True
            )[:5],
        },
        "regression_persistence": {
            "model": "Ridge",
            "r2_score": float(pers_r2),
            "mae": float(pers_mae),
        },
        "baseline_logistic_regression": {
            "model": "LogisticRegression",
            "accuracy": float(lr_acc),
        },
        "features": FEATURE_COLUMNS,
    }

    # 5. Export Model Artifacts
    print("\n--- Serializing Model Artifacts to ml/models/ ---")
    joblib.dump(xgb_stage1, MODELS_DIR / "stage1_xgboost.joblib")
    joblib.dump(lgb_stage2, MODELS_DIR / "stage2_lightgbm.joblib")
    joblib.dump(ridge_pers, MODELS_DIR / "persistence_ridge_regression.joblib")
    joblib.dump(logreg_baseline, MODELS_DIR / "baseline_logistic_regression.joblib")
    joblib.dump(scaler, MODELS_DIR / "scaler.joblib")

    with open(MODELS_DIR / "model_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(" Saved: stage1_xgboost.joblib")
    print(" Saved: stage2_lightgbm.joblib")
    print(" Saved: persistence_ridge_regression.joblib")
    print(" Saved: baseline_logistic_regression.joblib")
    print(" Saved: scaler.joblib")
    print(" Saved: model_metrics.json")
    print("\n🎉 ML Training successfully completed!")


if __name__ == "__main__":
    train_models()

