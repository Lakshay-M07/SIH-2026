# Day 11 API Contract Specification
*Prepared by Person B (Backend + Data Engineer) for Person C to record in `chat_context.md`.*

## Overview
This document defines the REST API contract between the Express backend, the ML inference pipeline, and the React GIS dashboard.

---

## Endpoints

### 1. Hotspot List (Paginated)
- **Endpoint**: `GET /api/fires`
- **Query Parameters**:
  - `page` (default: 1)
  - `per_page` (default: 50)
  - `class` (optional: `industrial` | `natural`)
  - `confidence_min` (optional: 0.0 - 1.0)
  - `state` (optional: state name)
  - `date_from` (optional: ISO 8601 string)
  - `date_to` (optional: ISO 8601 string)
- **Response Contract (Strict)**:
```json
{
  "data": [
    {
      "id": "fire_20260825_001",
      "latitude": 22.4707,
      "longitude": 70.0577,
      "acq_date": "2026-08-25",
      "acq_time": "0830",
      "satellite": "VIIRS_NPP",
      "instrument": "VIIRS",
      "confidence": "high",
      "bright_ti4": 367.2,
      "bright_ti5": 302.1,
      "frp": 45.8,
      "daynight": "D",
      "classification": "industrial",
      "class_probabilities": {
        "industrial": 0.92,
        "natural": 0.08
      },
      "persistent_source": true,
      "evidence": {
        "industrial_distance_m": 180,
        "refinery_distance_m": 420,
        "detections_last_30d": 17,
        "night_ratio": 0.78,
        "frp_persistence_score": 0.82,
        "satellite_classifier": "industrial 89%"
      }
    }
  ],
  "total": 839,
  "page": 1,
  "per_page": 50
}
```
*Note: `total` count must always be present so the frontend can display `Showing 5 of 839`.*

---

### 2. Single Hotspot Evidence
- **Endpoint**: `GET /api/fires/:id`
- **Response**: Full hotspot object including OSM spatial context, temporal history, and satellite imagery chip URL (or `"satellite_evidence": "unavailable"` fallback).

---

### 3. Current Active Hotspots
- **Endpoint**: `GET /api/fires/active`
- **Response**: Array of detections from the latest FIRMS NRT ingestion cycle (~3-hour latency clearly flagged).

---

### 4. Historical Hotspots
- **Endpoint**: `GET /api/fires/history`
- **Query Parameters**: `start_date`, `end_date`, `region`
- **Response**: Paginated or clustered historical events.

---

### 5. Aggregated Statistics
- **Endpoint**: `GET /api/statistics`
- **Response Contract**:
```json
{
  "total_active_hotspots": 124,
  "industrial_count": 38,
  "natural_count": 86,
  "persistent_sources_identified": 14,
  "top_states": [
    { "state": "Gujarat", "count": 28 },
    { "state": "Odisha", "count": 24 },
    { "state": "Chhattisgarh", "count": 19 }
  ],
  "last_firms_refresh": "2026-08-25T14:30:00Z",
  "latency_note": "Near Real-Time (≈3-hr delay)"
}
```

---

### 6. Run Inference
- **Endpoint**: `POST /api/predict`
- **Payload**: Hotspot record with FIRMS coordinates + temporal/spatial attributes.
- **Response**: Model version, prediction class, probabilities, and top explanation factors.
