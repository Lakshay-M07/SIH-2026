# Project Context

## August 25, 2026 (Day 1 — Person B: Backend & Data Engineer)
### Completed
- Installed Express dependencies (`express`, `cors`, `dotenv`) in `server/`.
- Built modular backend skeleton: `routes/`, `controllers/`, `services/`, `firebase/` directories.
- Implemented `GET /` root and `GET /api/health` endpoints (verified smoke test exit 0).
- Created Firebase safe initialization stub in `server/src/firebase/config.js` — server starts cleanly without credentials.
- Created `server/src/services/fire.service.js` boundary stub for Day 11 endpoints.
- Defined Day 11 API contract in `docs/api_contract_day11.md`.

## Architecture Decisions
### ML
- Primary model path: tabular geospatial classifier.
- Satellite imagery branch is an enhancement, not a dependency.
- Persistent thermal source is a separate event-level property.

### Backend
- Express API with Firebase / Firestore integration.

## Blockers
- None recorded at repository initialization.

## Task Allocation
- See `TEAM.md`.

---

## August 26, 2026 (Day 2 — Person B: Backend & Data Engineer)
### Completed
- Locked 13 FIRMS core fields in `ml/preprocessing/schema_firms.py` with alias support and `validate_firms_columns()` helper.
- Built `ml/preprocessing/loaders.py` with `load_firms_parquet()` — raises `FileNotFoundError` gracefully waiting on Person A handoff; rejects malformed files with `ValueError`.
- Created quarantined 3-row mock fixture at `ml/preprocessing/fixtures/mock_firms_sample.parquet` for offline testing.
- Created `server/src/services/dataAccessService.js` as a Day 2 smoke-test stub (temporary, not for production use).
- Updated `.env.example` with `FIRESTORE_PROJECT_ID` and `FIRESTORE_PRIVATE_KEY` placeholders.
- Verified exact test: `python -c "from ml.preprocessing.loaders import load_firms_parquet; load_firms_parquet('ml/preprocessing/fixtures/mock_firms_sample.parquet')"` exits 0.

---

## August 27, 2026 (Day 3 — Person B: Backend & Data Engineer)
### Completed
- Built `ml/preprocessing/spatial_join.py` — spatial distance feature utilities (vectorized haversine) for all 6 OSM categories: `distance_to_industrial_area`, `distance_to_power_plant`, `distance_to_refinery`, `distance_to_mine`, `distance_to_factory`, `distance_to_major_road`.
- Verified spatial join on mock fixture via `python -m ml.preprocessing.spatial_join --test-fixture`.
- Exported new utilities through `ml/preprocessing/__init__.py`.

### Notes
- `ml/ingestion/osm.py` stub left untouched — **Person A's deliverable** (Day 4).
- `ml/data/external/osm_industrial.geojson` is **Person C's output** (Day 5). Not produced by Person B.
- `spatial_join.py` will consume `osm_industrial.geojson` once Person C deposits it.

### Anti-Leakage Rule (Critical ML Integrity)
- **OSM infrastructure features are model inputs only, NEVER a target label source.** A hotspot overlapping or near an industrial facility does not automatically label it an industrial fire. Labels remain strictly derived from verified incident records (CPCB/State PCB) and manual spot-checking (`label_provenance.csv`).
