"""Model inference entry point for THERMOGRID Thermal Hotspot Intelligence.

Loads trained models from `ml/models/`:
- XGBoost (Stage 1: Environment Association)
- LightGBM (Stage 2: Behaviour Analysis)
- Ridge Regression (Continuous Persistence Score)

Provides batch and single-record inference APIs and CLI.
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"

import json
import sys
from pathlib import Path
import joblib
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = PROJECT_ROOT / "ml" / "models"

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


def haversine(lat1, lon1, lat2, lon2):
    """Haversine distance in kilometers."""
    r = 6371.0
    phi1 = np.radians(lat1)
    phi2 = np.radians(lat2)
    delta_phi = np.radians(lat2 - lat1)
    delta_lambda = np.radians(lon2 - lon1)
    a = (
        np.sin(delta_phi / 2.0) ** 2
        + np.cos(phi1) * np.cos(phi2) * np.sin(delta_lambda / 2.0) ** 2
    )
    return r * 2 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))


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

_MODELS = {}


def load_models():
    """Load models lazily into memory cache."""
    global _MODELS
    if _MODELS:
        return _MODELS

    xgb_path = MODELS_DIR / "stage1_xgboost.joblib"
    lgb_path = MODELS_DIR / "stage2_lightgbm.joblib"
    ridge_path = MODELS_DIR / "persistence_ridge_regression.joblib"
    scaler_path = MODELS_DIR / "scaler.joblib"

    if not xgb_path.exists() or not lgb_path.exists():
        raise FileNotFoundError(
            f"Model artifacts not found in {MODELS_DIR}. Run train.py first."
        )

    _MODELS = {
        "xgb_stage1": joblib.load(xgb_path),
        "lgb_stage2": joblib.load(lgb_path),
        "ridge_pers": joblib.load(ridge_path),
        "scaler": joblib.load(scaler_path),
    }
    return _MODELS


def _extract_features(records: list[dict]) -> np.ndarray:
    """Transform list of hotspot records into numeric feature array."""
    n = len(records)
    X = np.zeros((n, len(FEATURE_COLUMNS)), dtype=np.float32)

    for i, r in enumerate(records):
        lat = float(r.get("latitude", 30.0))
        lon = float(r.get("longitude", 75.0))
        bt4 = float(r.get("bright_ti4", r.get("brightness", 330.0)))
        bt5 = float(r.get("bright_ti5", 300.0))
        delta_bt = bt4 - bt5
        bt_ratio = bt4 / max(1.0, bt5)

        frp = float(r.get("frp", 2.0) or 0.0)
        log_frp = np.log1p(max(0.0, frp))

        scan = float(r.get("scan", 0.4) or 0.4)
        track = float(r.get("track", 0.38) or 0.38)
        pixel_area = scan * track

        conf_raw = r.get("confidence", "n")
        if isinstance(conf_raw, (int, float)):
            conf_num = float(conf_raw) if conf_raw <= 1.0 else float(conf_raw) / 100.0
        else:
            conf_map = {"l": 0.30, "n": 0.75, "h": 0.95}
            conf_num = conf_map.get(str(conf_raw).lower(), 0.70)

        is_day = 1.0 if str(r.get("daynight", "D")).upper() == "D" else 0.0
        hour = float(r.get("hour", 12.0))

        density = float(r.get("spatial_density_15km", 2.0))
        min_dist = float(r.get("min_neighbor_dist_km", 5.0))

        # Temporal features
        temporal_pass_count = float(r.get("temporal_pass_count", 2.0))
        cluster_time_span_days = float(r.get("cluster_time_span_days", 1.0))
        nocturnal_fraction = float(r.get("nocturnal_fraction", 0.0 if is_day == 1.0 else 1.0))

        # Industrial anchor distance
        dist_ind_m = float(r.get("dist_industrial_m", min([
            haversine(lat, lon, a_lat, a_lon) * 1000.0 for a_lat, a_lon, _ in INDUSTRIAL_ANCHORS
        ])))

        thermal_idx = (delta_bt / 30.0) * 0.40 + log_frp * 0.35 + conf_num * 0.25
        persistence_idx = (
            min(1.0, temporal_pass_count / 5.0) * 0.45
            + min(1.0, cluster_time_span_days / 7.0) * 0.25
            + nocturnal_fraction * 0.20
            + (1.0 if conf_num >= 0.7 else 0.0) * 0.10
        )

        X[i] = [
            lat,
            lon,
            bt4,
            bt5,
            delta_bt,
            bt_ratio,
            frp,
            log_frp,
            scan,
            track,
            pixel_area,
            conf_num,
            is_day,
            hour,
            density,
            min_dist,
            temporal_pass_count,
            cluster_time_span_days,
            nocturnal_fraction,
            dist_ind_m,
            thermal_idx,
            persistence_idx,
        ]

    return X


def predict_batch(records: list[dict]) -> list[dict]:
    """Run batch inference using XGBoost, LightGBM, and Ridge Regression."""
    if not records:
        return []

    models = load_models()
    X = _extract_features(records)
    X_scaled = models["scaler"].transform(X)

    # 1. XGBoost Stage 1 Predictions & Probabilities
    s1_probs = models["xgb_stage1"].predict_proba(X)
    s1_preds_idx = np.argmax(s1_probs, axis=1)

    # 2. LightGBM Stage 2 Predictions & Probabilities
    s2_probs = models["lgb_stage2"].predict_proba(X)
    s2_preds_idx = np.argmax(s2_probs, axis=1)

    # 3. Ridge Regression Persistence Predictions
    pers_preds = models["ridge_pers"].predict(X_scaled)
    pers_clipped = np.clip(pers_preds, 0.1, 0.98)

    results = []
    for i, r in enumerate(records):
        s1_class = STAGE1_CLASSES[s1_preds_idx[i]]
        s1_conf = float(s1_probs[i][s1_preds_idx[i]])
        s1_prob_list = [
            {"label": lbl, "probability": round(float(s1_probs[i][idx]), 3)}
            for idx, lbl in enumerate(STAGE1_CLASSES)
        ]

        s2_class = STAGE2_CLASSES[s2_preds_idx[i]]
        s2_conf = float(s2_probs[i][s2_preds_idx[i]])
        s2_prob_list = [
            {"label": lbl, "probability": round(float(s2_probs[i][idx]), 3)}
            for idx, lbl in enumerate(STAGE2_CLASSES)
        ]

        results.append({
            "id": r.get("id"),
            "stage1": {
                "stage": "environment",
                "prediction": s1_class,
                "confidence": round(s1_conf, 3),
                "probabilities": s1_prob_list,
            },
            "stage2": {
                "stage": "behaviour",
                "prediction": s2_class,
                "confidence": round(s2_conf, 3),
                "probabilities": s2_prob_list,
            },
            "persistence_score": round(float(pers_clipped[i]), 3),
            "ml_models": {
                "stage1": "XGBoost (XGBClassifier)",
                "stage2": "LightGBM (LGBMClassifier)",
                "regression": "Ridge (Linear Persistence Regressor)",
            },
        })

    return results


def predict_hotspot(record: dict) -> dict:
    """Single-record inference."""
    return predict_batch([record])[0]


def main() -> None:
    """CLI and IPC execution."""
    if "--stdin" in sys.argv:
        try:
            raw_in = sys.stdin.read()
            if not raw_in.strip():
                print(json.dumps([]))
                return
            input_data = json.loads(raw_in)
            if isinstance(input_data, dict):
                input_data = [input_data]
            preds = predict_batch(input_data)
            print(json.dumps(preds))
        except Exception as e:
            sys.stderr.write(f"Inference error: {e}\n")
            sys.exit(1)
        return

    print("Testing ML inference pipeline...")
    sample_records = [
        {
            "id": "sample_001",
            "latitude": 30.51386,
            "longitude": 73.90864,
            "bright_ti4": 335.57,
            "bright_ti5": 299.41,
            "frp": 2.87,
            "confidence": "n",
            "scan": 0.45,
            "track": 0.39,
            "daynight": "D",
        },
        {
            "id": "sample_002",
            "latitude": 31.10913,
            "longitude": 74.49384,
            "bright_ti4": 345.2,
            "bright_ti5": 299.84,
            "frp": 9.69,
            "confidence": "n",
            "scan": 0.48,
            "track": 0.4,
            "daynight": "D",
        },
    ]

    preds = predict_batch(sample_records)
    print(json.dumps(preds, indent=2))
    print("\n✅ Inference verification successful!")


if __name__ == "__main__":
    main()

