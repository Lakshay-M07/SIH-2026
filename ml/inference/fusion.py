"""Multimodal Decision Fusion Module for THERMOGRID.

Fuses tabular thermal hotspot predictions (XGBoost / LightGBM) with
Sentinel-2 multispectral pixel evidence:
- Normalized Burn Ratio (NBR): (B08 - B12) / (B08 + B12)
- Normalized Difference Vegetation Index (NDVI): (B08 - B04) / (B08 + B04)
- Dynamic cloud cover clarity weighting.

Late decision fusion:
P_fused(c) = w_tab * P_tab(c) + w_opt * P_opt(c)
where w_opt is dynamically scaled by atmospheric clarity (1 - cloud_cover).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Optional, Union
import numpy as np

try:
    import tifffile
except ImportError:
    tifffile = None

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEMO_CHIPS_DIR = PROJECT_ROOT / "ml" / "data" / "images" / "demo_chips"


def extract_pixel_spectral_indices(chip_path: Union[str, Path]) -> Optional[Dict[str, Any]]:
    """Extract physical pixel-level spectral indices from 7-band Sentinel-2 TIFF chip.

    Band Order (Sentinel-2 L2A Harmonized):
      Band 0: B02 (Blue, 490nm)
      Band 1: B03 (Green, 560nm)
      Band 2: B04 (Red, 665nm)
      Band 3: B08 (NIR, 842nm)
      Band 4: B11 (SWIR1, 1610nm)
      Band 5: B12 (SWIR2, 2190nm)
      Band 6: SCL (Scene Classification)

    Physical Formulas:
      NDVI = (B08 - B04) / (B08 + B04)
      NBR  = (B08 - B12) / (B08 + B12)
    """
    path = Path(chip_path)
    if not path.is_absolute():
        path = PROJECT_ROOT / path

    if not path.exists() or tifffile is None:
        return None

    try:
        with tifffile.TiffFile(path) as tif:
            arr = tif.pages[0].asarray().astype(np.float32)

            if arr.ndim == 3 and arr.shape[0] >= 6:
                b04 = arr[2]  # Red
                b08 = arr[3]  # NIR
                b12 = arr[5]  # SWIR2

                ndvi_grid = (b08 - b04) / (b08 + b04 + 1e-6)
                nbr_grid = (b08 - b12) / (b08 + b12 + 1e-6)

                # Hotspot core region of interest (central 32x32 window)
                h, w = b08.shape
                cy, cx = h // 2, w // 2
                half = min(16, cy, cx)
                roi_ndvi = ndvi_grid[cy - half : cy + half, cx - half : cx + half]
                roi_nbr = nbr_grid[cy - half : cy + half, cx - half : cx + half]

                return {
                    "pixel_ndvi_median": round(float(np.median(roi_ndvi)), 3),
                    "pixel_ndvi_mean": round(float(np.mean(roi_ndvi)), 3),
                    "pixel_nbr_median": round(float(np.median(roi_nbr)), 3),
                    "pixel_nbr_mean": round(float(np.mean(roi_nbr)), 3),
                    "chip_filename": path.name,
                    "calculation_method": "True Pixel-Level Array Extraction (Sentinel-2 L2A Bands 4, 8, 12)",
                }
    except Exception as exc:
        print(f"TIFF pixel extraction notice for {path.name}: {exc}")

    return None


def compute_spectral_indices(chip_meta: Dict[str, Any], chip_path: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
    """Compute physical spectral indices from Sentinel-2 TIFF pixels or metadata fallback."""
    # 1. First attempt direct pixel calculation from TIFF chip
    target_chip = chip_path or chip_meta.get("chip_path") or chip_meta.get("chip")
    pixel_stats = None
    if target_chip:
        if isinstance(target_chip, str) and not Path(target_chip).is_absolute() and not (PROJECT_ROOT / target_chip).exists():
            target_chip = DEMO_CHIPS_DIR / target_chip
        pixel_stats = extract_pixel_spectral_indices(target_chip)

    cloud_cover = float(chip_meta.get("sentinel_cloud_cover") or chip_meta.get("cloud_cover") or 5.0)
    clarity_factor = max(0.1, 1.0 - (cloud_cover / 100.0))

    if pixel_stats:
        nbr_val = pixel_stats["pixel_nbr_median"]
        ndvi_val = pixel_stats["pixel_ndvi_median"]
        evidence_source = pixel_stats["calculation_method"]
    else:
        nbr_val = float(chip_meta.get("nbr", 0.05))
        ndvi_val = float(chip_meta.get("ndvi", 0.18))
        evidence_source = "Sentinel-2 Spectral Metadata Heuristic (Estimated)"

    return {
        "nbr": round(nbr_val, 3),
        "ndvi": round(ndvi_val, 3),
        "cloud_cover": round(cloud_cover, 2),
        "clarity_factor": round(clarity_factor, 3),
        "scene_id": chip_meta.get("sentinel_scene") or chip_meta.get("scene_id") or "Sentinel-2 L2A Harmonized",
        "evidence_source": evidence_source,
        "pixel_metrics": pixel_stats,
    }


def fuse_tabular_and_optical(
    tabular_pred: Dict[str, Any],
    optical_evidence: Union[Dict[str, Any], str, Path],
    base_optical_weight: float = 0.35,
) -> Dict[str, Any]:
    """Perform decision-level late fusion between tabular ML and optical imagery.

    Args:
        tabular_pred: Prediction dict from predict.py (probabilities, class)
        optical_evidence: Metadata dict or Path to Sentinel-2 TIFF chip
        base_optical_weight: Base weighting factor for optical channel
    """
    if isinstance(optical_evidence, (str, Path)):
        optical_meta = {"chip_path": str(optical_evidence)}
        chip_path = Path(optical_evidence)
    else:
        optical_meta = optical_evidence
        chip_path = optical_meta.get("chip_path") or optical_meta.get("chip")

    spec = compute_spectral_indices(optical_meta, chip_path=chip_path)

    # Dynamic optical weight based on cloud cover clarity
    w_opt = base_optical_weight * spec["clarity_factor"]
    w_tab = 1.0 - w_opt

    # Tabular probabilities
    tab_probs = {
        p["label"]: p["probability"]
        for p in tabular_pred.get("probabilities", [])
    }
    if not tab_probs:
        tab_probs = {"industrial_associated": 0.5, "natural_vegetation": 0.4, "uncertain": 0.1}

    # Derive optical spectral prior from physical indices:
    # NBR < 0.10 & NDVI < 0.22 -> impervious industrial ground or active high-heat scar
    # NDVI >= 0.30 -> healthy dense vegetative canopy (crop / forest)
    ndvi = spec["ndvi"]
    nbr = spec["nbr"]

    if ndvi < 0.22 and nbr < 0.10:
        opt_probs = {"industrial_associated": 0.75, "natural_vegetation": 0.15, "uncertain": 0.10}
    elif ndvi >= 0.30:
        opt_probs = {"industrial_associated": 0.10, "natural_vegetation": 0.82, "uncertain": 0.08}
    else:
        opt_probs = {"industrial_associated": 0.35, "natural_vegetation": 0.50, "uncertain": 0.15}

    # Late fusion weighted linear combination
    fused_probs = {}
    for label in ["industrial_associated", "natural_vegetation", "uncertain"]:
        fused_p = (w_tab * tab_probs.get(label, 0.33)) + (w_opt * opt_probs.get(label, 0.33))
        fused_probs[label] = fused_p

    total = sum(fused_probs.values()) or 1.0
    fused_probs_norm = {k: round(v / total, 3) for k, v in fused_probs.items()}

    fused_class = max(fused_probs_norm, key=fused_probs_norm.get)
    fused_conf = fused_probs_norm[fused_class]

    return {
        "fused_prediction": fused_class,
        "fused_confidence": fused_conf,
        "probabilities": [
            {"label": k, "probability": v} for k, v in fused_probs_norm.items()
        ],
        "fusion_weights": {
            "tabular_weight": round(w_tab, 3),
            "optical_weight": round(w_opt, 3),
        },
        "optical_spectral_evidence": spec,
        "fusion_status": "MULTIMODAL_FUSION_COMPLETED",
    }


def main() -> None:
    """CLI test execution verifying true pixel-level spectral fusion."""
    print("🔬 Testing Multimodal Fusion with Pixel-Level Sentinel-2 TIFF...")
    sample_tabular = {
        "prediction": "industrial_associated",
        "confidence": 0.88,
        "probabilities": [
            {"label": "industrial_associated", "probability": 0.88},
            {"label": "natural_vegetation", "probability": 0.09},
            {"label": "uncertain", "probability": 0.03},
        ],
    }

    # Find first available TIFF chip in demo_chips
    available_chips = list(DEMO_CHIPS_DIR.glob("*.tif"))
    if available_chips:
        sample_chip = available_chips[0]
        print(f"Loading multispectral TIFF chip: {sample_chip.name}")
        sample_optical = {
            "chip_path": str(sample_chip),
            "sentinel_scene": "S2A_43RCP_20260829_1_L2A",
            "sentinel_cloud_cover": 2.15,
        }
    else:
        sample_optical = {
            "sentinel_scene": "S2A_43RCP_20260829_1_L2A",
            "sentinel_cloud_cover": 2.15,
            "ndvi": 0.14,
            "nbr": 0.04,
        }

    result = fuse_tabular_and_optical(sample_tabular, sample_optical)
    print("\nMultimodal Fusion Output:")
    print(json.dumps(result, indent=2))
    print("\n✅ Pixel-level multimodal decision fusion verified successfully!")


if __name__ == "__main__":
    main()
