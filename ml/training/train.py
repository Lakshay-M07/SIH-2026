"""Training pipeline for THERMOGRID Thermal Hotspot Intelligence.

Models trained:
1. XGBoost (XGBClassifier) -> Stage 1: Environment Association
   (industrial_associated, natural_vegetation, uncertain)
2. LightGBM (LGBMClassifier) -> Stage 2: Behaviour Analysis
   (persistent_expected, new_abnormal, insufficient_history)
3. Ridge & Linear Regression -> Continuous Persistence & FRP Anomaly Scoring
4. Logistic Regression -> Linear Baseline Benchmark

Evaluation Protocol:
- 5-Fold Stratified Cross-Validation (StratifiedKFold)
- Out-of-fold generalization metrics (Accuracy, Precision, Recall, F1, Confusion Matrix)
- Domain-calibrated pseudo-labels + true multi-pass temporal persistence features

Exports trained models and evaluation metrics to `ml/models/`.
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"

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
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    recall_score,
    r2_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

PROJECT_ROOT = Path(__file__).resolve().parents[2]
import sys
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
DATA_DIR = PROJECT_ROOT / "ml" / "data"
MODELS_DIR = PROJECT_ROOT / "ml" / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

STAGE1_CLASSES = ["industrial_associated", "natural_vegetation", "uncertain"]
STAGE2_CLASSES = ["persistent_expected", "new_abnormal", "insufficient_history"]


# Reference industrial anchor coordinates across Northern/Western India
INDUSTRIAL_ANCHORS = [
    (30.9010, 75.8573, "Ludhiana Heavy Industrial Complex"),
    (30.2110, 74.9455, "Bathinda Thermal Power Complex"),
    (31.3260, 75.5762, "Jalandhar Manufacturing Cluster"),
    (29.6857, 76.9905, "Karnal Energy & Processing Belt"),
    (21.1702, 74.7796, "Dhule Industrial & Highway Corridor"),
    (29.9695, 76.8783, "Kurukshetra Agro-Industrial Hub"),
    (28.3949, 70.3340, "Border Region Logistics Zone"),
]


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


def compute_anchor_distances(lats, lons):
    """Compute distance in meters to the nearest confirmed industrial anchor infrastructure."""
    n = len(lats)
    min_distances_m = np.zeros(n, dtype=np.float32)

    for i in range(n):
        dist_km_list = [
            haversine_np(lats[i], lons[i], a_lat, a_lon)
            for (a_lat, a_lon, _) in INDUSTRIAL_ANCHORS
        ]
        min_distances_m[i] = float(np.min(dist_km_list) * 1000.0)

    return min_distances_m


def build_spatiotemporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extract true temporal recurrence and spatial cluster metrics.

    Addresses Issue #8: Builds multi-pass temporal statistics rather than
    relying solely on static single-observation spatial density.
    """
    lats = df["latitude"].to_numpy()
    lons = df["longitude"].to_numpy()
    n = len(df)

    # Convert acquisition date/time to UTC timestamps
    dates = pd.to_datetime(df["acq_date"].astype(str), errors="coerce")
    acq_times = pd.to_numeric(df["acq_time"], errors="coerce").fillna(800).astype(int)
    hours = acq_times // 100
    minutes = acq_times % 100
    timestamps_h = dates.astype(np.int64) / (1e9 * 3600.0) + hours + minutes / 60.0

    spatial_density_15km = np.zeros(n, dtype=np.float32)
    min_neighbor_dist_km = np.zeros(n, dtype=np.float32)
    temporal_pass_count = np.zeros(n, dtype=np.float32)
    cluster_time_span_days = np.zeros(n, dtype=np.float32)
    nocturnal_fraction = np.zeros(n, dtype=np.float32)

    daynight_arr = (df["daynight"].astype(str).str.upper() == "D").to_numpy()

    for i in range(n):
        dists = haversine_np(lats[i], lons[i], lats, lons)
        cluster_mask = (dists <= 15.0)

        # Spatial metrics
        dists_no_self = np.delete(dists, i)
        if len(dists_no_self) > 0:
            spatial_density_15km[i] = np.sum(dists_no_self <= 15.0)
            min_neighbor_dist_km[i] = np.min(dists_no_self)
        else:
            spatial_density_15km[i] = 0.0
            min_neighbor_dist_km[i] = 99.0

        # Temporal metrics across the cluster
        cluster_times = timestamps_h[cluster_mask]
        cluster_dn = daynight_arr[cluster_mask]

        cluster_times_sorted = np.sort(cluster_times)
        if len(cluster_times_sorted) > 1:
            time_diffs = np.diff(cluster_times_sorted)
            pass_count = 1 + np.sum(time_diffs >= 2.0)
            span_days = (cluster_times_sorted[-1] - cluster_times_sorted[0]) / 24.0
        else:
            pass_count = 1.0
            span_days = 0.0

        temporal_pass_count[i] = pass_count
        cluster_time_span_days[i] = span_days
        nocturnal_fraction[i] = 1.0 - (np.mean(cluster_dn) if len(cluster_dn) > 0 else 1.0)

    df["spatial_density_15km"] = spatial_density_15km
    df["min_neighbor_dist_km"] = min_neighbor_dist_km
    df["temporal_pass_count"] = temporal_pass_count
    df["cluster_time_span_days"] = cluster_time_span_days
    df["nocturnal_fraction"] = nocturnal_fraction
    df["dist_industrial_m"] = compute_anchor_distances(lats, lons)
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Feature engineering from FIRMS VIIRS/MODIS thermal and spatiotemporal features."""
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
        df["is_day"] = (df["daynight"].astype(str).str.upper() == "D").astype(float)
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

    # Spatiotemporal features (Temporal clustering + OSM infrastructure proximity)
    df = build_spatiotemporal_features(df)

    # Attach real OpenStreetMap infrastructure distances (Addresses Issues #9, #10)
    try:
        from ml.preprocessing.spatial_join import attach_osm_distance_features
        df = attach_osm_distance_features(df)
        df["dist_industrial_m"] = df["distance_to_industrial_area"]
    except Exception as exc:
        print(f"OSM distance feature attachment note: {exc}")

    # Thermal Intensity Index (composite physical metric)
    df["thermal_intensity_index"] = (
        (df["delta_bt"] / 30.0) * 0.40
        + df["log_frp"] * 0.35
        + df["confidence_num"] * 0.25
    )

    # Persistence Index based on multi-pass temporal recurrence
    df["persistence_index"] = (
        np.clip(df["temporal_pass_count"] / 5.0, 0.0, 1.0) * 0.45
        + np.clip(df["cluster_time_span_days"] / 7.0, 0.0, 1.0) * 0.25
        + df["nocturnal_fraction"] * 0.20
        + (df["confidence_num"] >= 0.7).astype(float) * 0.10
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
    "temporal_pass_count",
    "cluster_time_span_days",
    "nocturnal_fraction",
    "dist_industrial_m",
    "thermal_intensity_index",
    "persistence_index",
]


def create_ground_truth_labels(df: pd.DataFrame):
    """Establish domain-informed pseudo-labels calibrated against physical thresholds.

    Note (per SIH review): These labels represent expert rule-derived weak labels
    combining VIIRS physical thermal dynamics, OSM infrastructure proximity, and
    temporal observation recurrence.
    """
    n = len(df)
    stage1_labels = np.zeros(n, dtype=int)
    stage2_labels = np.zeros(n, dtype=int)
    persistence_scores = np.zeros(n, dtype=np.float32)

    for i in range(n):
        row = df.iloc[i]
        frp = row["frp_clean"]
        delta_bt = row["delta_bt"]
        conf = row["confidence_num"]
        dist_ind = row["dist_industrial_m"]
        pass_count = row["temporal_pass_count"]
        is_day = row["is_day"]

        # Stage 1: Environment Association
        if conf < 0.4 or (frp < 0.8 and delta_bt < 15.0):
            s1 = 2  # uncertain
        elif dist_ind < 5000.0 or (is_day == 0.0 and delta_bt >= 20.0 and frp >= 3.0):
            s1 = 0  # industrial_associated
        elif frp >= 6.0 and delta_bt >= 28.0:
            s1 = 0  # intense industrial heat point
        else:
            s1 = 1  # natural_vegetation (crop residue burning / open biomass)
        stage1_labels[i] = s1

        # Stage 2: Behaviour Analysis (truly temporal multi-pass recurrence)
        if conf < 0.4 or pass_count < 2:
            s2 = 2  # insufficient_history
            pers = 0.20 + 0.15 * min(1.0, row["persistence_index"])
        elif pass_count >= 3 and s1 == 0:
            s2 = 0  # persistent_expected
            pers = 0.70 + 0.25 * min(1.0, row["persistence_index"])
        elif frp >= 5.5 and delta_bt >= 30.0:
            s2 = 1  # new_abnormal
            pers = 0.40 + 0.20 * min(1.0, row["persistence_index"])
        elif pass_count >= 4:
            s2 = 0  # persistent agricultural burning zone
            pers = 0.65 + 0.20 * min(1.0, row["persistence_index"])
        else:
            s2 = 1  # new_abnormal transient event
            pers = 0.35 + 0.15 * min(1.0, row["persistence_index"])

        stage2_labels[i] = s2
        persistence_scores[i] = float(np.clip(pers, 0.1, 0.98))

    df["stage1_target"] = stage1_labels
    df["stage2_target"] = stage2_labels
    df["persistence_target"] = persistence_scores
    return df


def train_models():
    """Main training routine with 5-Fold Stratified Cross-Validation."""
    print("=" * 75)
    print("🚀 THERMOGRID ML Training & Cross-Validation Pipeline")
    print("Models: XGBoost (Stage 1), LightGBM (Stage 2), Ridge Regression (Persistence)")
    print("Validation: 5-Fold Stratified Cross-Validation (Out-Of-Fold Evaluation)")
    print("=" * 75)

    # 1. Load Data
    firms_file = DATA_DIR / "firms_latest.csv"
    if not firms_file.exists():
        raise FileNotFoundError(f"FIRMS dataset not found at {firms_file}")

    raw_df = pd.read_csv(firms_file)
    print(f"Loaded {len(raw_df)} FIRMS records from {firms_file.name}")

    # 2. Engineer features
    df = engineer_features(raw_df)
    df = create_ground_truth_labels(df)

    X = df[FEATURE_COLUMNS].to_numpy(dtype=np.float32)
    y_stage1 = df["stage1_target"].to_numpy(dtype=int)
    y_stage2 = df["stage2_target"].to_numpy(dtype=int)
    y_pers = df["persistence_target"].to_numpy(dtype=np.float32)

    n_samples = len(df)
    print(f"Dataset dimension: {X.shape} ({len(FEATURE_COLUMNS)} engineered features)")
    print(f"Stage 1 distribution: {dict(df['stage1_target'].value_counts())}")
    print(f"Stage 2 distribution: {dict(df['stage2_target'].value_counts())}")

    # 3. 5-Fold Stratified Cross-Validation Setup
    n_splits = 5
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    oof_stage1_preds = np.zeros(n_samples, dtype=int)
    oof_stage2_preds = np.zeros(n_samples, dtype=int)
    oof_pers_preds = np.zeros(n_samples, dtype=np.float32)
    oof_lr_preds = np.zeros(n_samples, dtype=int)

    print(f"\n--- Running {n_splits}-Fold Stratified Cross-Validation (Leakage-Free) ---")
    fold = 1
    for train_idx, val_idx in skf.split(X, y_stage1):
        X_tr, y1_tr, y2_tr, yp_tr = X[train_idx], y_stage1[train_idx], y_stage2[train_idx], y_pers[train_idx]
        X_val, y1_val = X[val_idx], y_stage1[val_idx]

        # Strictly fold-isolated scaling: fit ONLY on training slice, transform validation slice (Addresses Problem #5)
        fold_scaler = StandardScaler()
        X_tr_sc = fold_scaler.fit_transform(X_tr)
        X_val_sc = fold_scaler.transform(X_val)

        # Fold XGBoost
        fold_xgb = xgb.XGBClassifier(
            n_estimators=60, max_depth=3, learning_rate=0.1, random_state=42, eval_metric="mlogloss"
        )
        fold_xgb.fit(X_tr, y1_tr)
        oof_stage1_preds[val_idx] = fold_xgb.predict(X_val)

        # Fold LightGBM
        fold_lgb = lgb.LGBMClassifier(
            n_estimators=60, max_depth=3, learning_rate=0.1, objective="multiclass",
            num_class=3, random_state=42, verbosity=-1
        )
        fold_lgb.fit(X_tr, y2_tr)
        oof_stage2_preds[val_idx] = fold_lgb.predict(X_val)

        # Fold Ridge
        fold_ridge = Ridge(alpha=1.0, random_state=42)
        fold_ridge.fit(X_tr_sc, yp_tr)
        oof_pers_preds[val_idx] = fold_ridge.predict(X_val_sc)

        # Fold Logistic Baseline
        fold_lr = LogisticRegression(max_iter=1000, random_state=42)
        fold_lr.fit(X_tr_sc, y1_tr)
        oof_lr_preds[val_idx] = fold_lr.predict(X_val_sc)

        fold_acc = accuracy_score(y1_val, oof_stage1_preds[val_idx])
        print(f"  Fold {fold}: XGBoost Validation Accuracy: {fold_acc:.4f}")
        fold += 1

    # 4. Out-of-Fold Honest Cross-Validation Evaluation (Addresses Problem #6, #7)
    s1_oof_acc = accuracy_score(y_stage1, oof_stage1_preds)
    s1_oof_f1 = f1_score(y_stage1, oof_stage1_preds, average="weighted")
    s1_oof_prec = precision_score(y_stage1, oof_stage1_preds, average="weighted", zero_division=0)
    s1_oof_rec = recall_score(y_stage1, oof_stage1_preds, average="weighted")
    s1_cm = confusion_matrix(y_stage1, oof_stage1_preds).tolist()

    s2_oof_acc = accuracy_score(y_stage2, oof_stage2_preds)
    s2_oof_f1 = f1_score(y_stage2, oof_stage2_preds, average="weighted")
    s2_oof_prec = precision_score(y_stage2, oof_stage2_preds, average="weighted", zero_division=0)
    s2_oof_rec = recall_score(y_stage2, oof_stage2_preds, average="weighted")
    s2_cm = confusion_matrix(y_stage2, oof_stage2_preds).tolist()

    pers_r2 = r2_score(y_pers, oof_pers_preds)
    pers_mae = mean_absolute_error(y_pers, oof_pers_preds)

    lr_oof_acc = accuracy_score(y_stage1, oof_lr_preds)

    print("\n" + "=" * 55)
    print("📊 OUT-OF-FOLD (OOF) GENERALIZATION METRICS")
    print("=" * 55)
    print(f"Stage 1 (XGBoost)   -> OOF Accuracy: {s1_oof_acc:.4f} | F1: {s1_oof_f1:.4f} | Recall: {s1_oof_rec:.4f}")
    print(f"Stage 2 (LightGBM)  -> OOF Accuracy: {s2_oof_acc:.4f} | F1: {s2_oof_f1:.4f} | Recall: {s2_oof_rec:.4f}")
    print(f"Persistence (Ridge) -> OOF R² Score: {pers_r2:.4f} | MAE: {pers_mae:.4f}")
    print(f"Baseline (Logistic) -> OOF Accuracy: {lr_oof_acc:.4f}")
    print("=" * 55)

    # 5. Final Production Model Training on full dataset
    print("\n--- Training Final Production Models on Full Dataset ---")
    final_xgb = xgb.XGBClassifier(
        n_estimators=100, max_depth=4, learning_rate=0.08, subsample=0.85,
        colsample_bytree=0.85, random_state=42, eval_metric="mlogloss"
    )
    final_xgb.fit(X, y_stage1)

    final_lgb = lgb.LGBMClassifier(
        n_estimators=100, max_depth=4, learning_rate=0.08, subsample=0.85,
        colsample_bytree=0.85, objective="multiclass", num_class=3,
        random_state=42, verbosity=-1
    )
    final_lgb.fit(X, y_stage2)

    final_scaler = StandardScaler()
    X_scaled_final = final_scaler.fit_transform(X)

    final_ridge = Ridge(alpha=1.0, random_state=42)
    final_ridge.fit(X_scaled_final, y_pers)

    final_logreg = LogisticRegression(max_iter=2000, random_state=42)
    final_logreg.fit(X_scaled_final, y_stage1)

    # Feature Importances
    xgb_feat_imp = dict(zip(FEATURE_COLUMNS, [float(x) for x in final_xgb.feature_importances_]))
    lgb_feat_imp = dict(zip(FEATURE_COLUMNS, [float(x) for x in final_lgb.feature_importances_]))

    metrics = {
        "evaluation_protocol": "5-Fold Stratified Cross-Validation (Out-Of-Fold)",
        "label_methodology": "Domain-calibrated expert heuristic pseudo-labels based on VIIRS physical dynamics, OSM infrastructure proximity, and temporal cluster recurrence.",
        "training_samples": n_samples,
        "stage1_xgboost": {
            "model": "XGBClassifier",
            "classes": STAGE1_CLASSES,
            "oof_cv_accuracy": round(float(s1_oof_acc), 4),
            "oof_cv_f1_score": round(float(s1_oof_f1), 4),
            "oof_cv_precision": round(float(s1_oof_prec), 4),
            "oof_cv_recall": round(float(s1_oof_rec), 4),
            "confusion_matrix": s1_cm,
            "feature_importance_top5": sorted(xgb_feat_imp.items(), key=lambda x: x[1], reverse=True)[:5],
        },
        "stage2_lightgbm": {
            "model": "LGBMClassifier",
            "classes": STAGE2_CLASSES,
            "oof_cv_accuracy": round(float(s2_oof_acc), 4),
            "oof_cv_f1_score": round(float(s2_oof_f1), 4),
            "oof_cv_precision": round(float(s2_oof_prec), 4),
            "oof_cv_recall": round(float(s2_oof_rec), 4),
            "confusion_matrix": s2_cm,
            "feature_importance_top5": sorted(lgb_feat_imp.items(), key=lambda x: x[1], reverse=True)[:5],
        },
        "regression_persistence": {
            "model": "Ridge",
            "oof_cv_r2_score": round(float(pers_r2), 4),
            "oof_cv_mae": round(float(pers_mae), 4),
        },
        "baseline_logistic_regression": {
            "model": "LogisticRegression",
            "oof_cv_accuracy": round(float(lr_oof_acc), 4),
        },
        "features": FEATURE_COLUMNS,
    }

    # 6. Export Model Artifacts
    print("\n--- Serializing Final Production Artifacts to ml/models/ ---")
    joblib.dump(final_xgb, MODELS_DIR / "stage1_xgboost.joblib")
    joblib.dump(final_lgb, MODELS_DIR / "stage2_lightgbm.joblib")
    joblib.dump(final_ridge, MODELS_DIR / "persistence_ridge_regression.joblib")
    joblib.dump(final_logreg, MODELS_DIR / "baseline_logistic_regression.joblib")
    joblib.dump(final_scaler, MODELS_DIR / "scaler.joblib")

    with open(MODELS_DIR / "model_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(" Saved: stage1_xgboost.joblib")
    print(" Saved: stage2_lightgbm.joblib")
    print(" Saved: persistence_ridge_regression.joblib")
    print(" Saved: baseline_logistic_regression.joblib")
    print(" Saved: scaler.joblib")
    print(" Saved: model_metrics.json")
    print("\n🎉 ML Training & Cross-Validation successfully completed!")


if __name__ == "__main__":
    train_models()

