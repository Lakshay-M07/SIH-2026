# THERMOGRID — proposed backend API contract v1

**Status: proposed, not implemented in this repository.** The only existing Express route is a root initialization response; it does not satisfy these contracts. This document is the integration handoff for a future FastAPI backend. Python ML modules remain on the server. Root architecture docs describe an older Express/Firebase plan; this interface follows the requested FastAPI/PostgreSQL/PostGIS direction without changing those modules.

## Configuration and envelope

Set `VITE_API_BASE_URL` to the public API root, including any path prefix such as `/api`. The frontend appends the paths below. No endpoint URL appears in a presentation component. Missing configuration immediately shows an unavailable state; there is no data fallback. Use HTTPS outside local development and allow the exact frontend origin in CORS. Requests have a 15-second timeout, canceled superseded reads, no automatic mutation retries, and runtime schema validation.

All JSON successes use `{ "data": T, "meta"?: { "total": number, "page": number, "page_size": number, "truncated"?: boolean } }`. Paged lists MUST include `meta` to support forward pagination. Do not return `null` for lists. Use `data: []` for an empty successful result. Errors use non-2xx HTTP status codes; internal details must not expose secrets. Download endpoints return binary, not an envelope. A malformed success body raises `INVALID_RESPONSE` in the UI.

Canonical machine-readable types/schemas are `client/src/types/index.ts`; services are `client/src/services/`. ISO 8601 UTC timestamps are required. `confidence`, stage probabilities and `persistence_score` are fractions in [0,1]; the browser only formats percentages. FRP is MW; brightness temperature is kelvin; coordinates use WGS84. Fields nullable in the schema must be present as null if unavailable; optional fields may be omitted.

## Shared filter query parameters

`from`, `to` (inclusive UTC calendar dates YYYY-MM-DD), `type`, `confidence_min`, `state`, `district`, `sensor`, `risk`, `persistence_min`, `classification`, `search`, `bbox` (`west,south,east,north`), `page` (1-based), `page_size`, `max_points`.

All supplied filter parameters apply conjunctively. `classification` matches either stage's prediction. Type: `industrial_fire | natural_fire | persistent_source`. Risk: `low | medium | high | critical`. Backend region/sensor values are opaque strings used consistently by `/regions` and list endpoints. The server validates dates, coordinates and bounds, rejects unsafe/unbounded requests, uses geospatial indexes, filters server-side, and returns stable ordering (latest timestamp, then ID). State changes reset district in the UI. Date range is shared by live map, dashboard, analytics and history; historical replay never downloads unlimited history.

## Health and capabilities

`GET /health` → `data: { status: "ok" | "degraded", version: string, capabilities: { live_polling: boolean, analyst_review: boolean, reports: boolean, historical_replay: boolean, gis_layers: boolean } }`.

No status is inferred from an unrelated successful response. Live polling requires status `ok`, `live_polling: true` and the local polling preference. Health checks every 30 seconds only when a base URL is configured; interval is 15/30/60/120 seconds for active data queries. Missing/failed health stops live polling. A returned health status does not independently confirm NASA/OSM ingestion freshness. Future SSE/WebSocket invalidations can replace the polling transport without changing presentation components.

## Hotspots

`GET /hotspots` → `data: Hotspot[]` + pagination meta.

For map queries, include `bbox`, `page=1`, `page_size=5000`, `max_points=5000`; enforce these ceilings server-side. Return `meta.truncated=true` when finer zoom is required. Browser clustering only applies to this bounded subset. Million-record support requires backend vector tiles or server clusters, not lifting client limits. Table queries use `page_size=10`; map search uses `search` and `page_size=8`.

`Hotspot`: `id: string`, `latitude: number`, `longitude: number`, `type: source enum`, `risk: risk enum`, `confidence: fraction`, `location: string`, `state: string`, `district: string`, `sensor: string`, `frp: number|null`, `brightness_temperature: number|null`, `persistence_score: fraction|null`, `first_detected: timestamp`, `last_detected: timestamp`.

`GET /hotspots/{id}` → `data: HotspotDetail`. The detail extends Hotspot with:
- `classifications: Classification[]` (one per environment/behaviour stage when available; missing stage is honestly shown as unavailable).
- `evidence: Evidence[]`.
- `satellite_evidence: SatelliteEvidence[]`.
- `detection_history: { timestamp, frp: number|null }[]` (max 500; backend must aggregate or bound).
- `verification: { classification: source enum, verifier: string, verified_at: timestamp, note: string } | null`.

`Classification`: `stage: "environment"|"behaviour"`, `prediction: string`, `confidence: fraction`, `probabilities: { label: string, probability: fraction }[]`, `evidence: Evidence[]`, optional nullable `persistence_score`, optional nullable `temporal_summary: string`.

Stage 1 prediction labels: `industrial_associated`, `natural_vegetation`, `uncertain`. Stage 2: `persistent_expected`, `new_abnormal`, `insufficient_history`. Final detection type must be supplied by the backend; the browser never derives it from probabilities. Probability normalization and class validity are backend responsibilities; the client rejects out-of-range numeric values.

`Evidence`: `id`, `label`, `value`, `source` (all strings), `kind: "model_input"|"verified"`, optional nullable `verifier`, `verified_at`, `confidence_note`. OSM, facility data and Sentinel context MUST be `model_input` unless independently verified with traceable provenance. The frontend never upgrades evidence to ground truth.

`SatelliteEvidence`: `id`, `image_url` (HTTPS), `acquisition_date`, `source`, `imagery_type`, `role: "thermal_detection"|"optical_context"`. These are actual backend image references, never stock previews. Sentinel-2 belongs to optical/contextual evidence, not primary thermal hotspot detection.

`GET /regions?state=<optional>` → `data: { states: string[], districts: string[], sensors: string[] }`. With state specified, return only its available districts. Populate with actual backend coverage rather than fictional locations. Failure leaves all-region choices usable and clearly marks missing options.

## Analytics

`GET /analytics/summary` with shared filters → `data: { total_detections, industrial_fires, natural_fires, persistent_sources, high_risk_events, updated_at }`. Counts are nonnegative integers. No client calculation of database-wide totals.

`GET /analytics/trends` with shared filters + `max_points=90` → `data: { fire_trend, distribution, top_states, confidence, persistence, frp, districts }`.

Each series item has `label: string` plus numeric aggregate `value` or the `industrial`, `natural`, `persistent` values used by fire-trend charts. Series must use consistent UTC bins/units and labeled aggregates. Max lengths: fire_trend/persistence/frp 366, distribution 20, top_states 36, confidence 100, districts 100. Empty arrays are valid, not zero-valued observations. These chart values are all server aggregates. Future contract versions should include aggregation/unit metadata explicitly.

## History and replay

`GET /historical` → same bounded Hotspot[] envelope/pagination as `/hotspots`, with shared filters and required practical date limits enforced server-side. Map uses bbox/5000 cap; table uses 10-record pages.

`GET /historical/replay` with shared filters, `max_frames=120`, `max_points=5000` → `data: { id: string, created_at: timestamp, frames: { timestamp, hotspots: Hotspot[] }[] }`. Enforce max 120 chronologically ordered frames and max 5,000 observations each, with a reasonable server response-size ceiling. Authentication/authorization applies as needed.

Offline JSON export contains exactly this replay `data` object, not invented frames. Imports are capped at 20 MB and validated with the same schema. Imported files are explicitly marked offline and are not independently provenance-verified. Replay controls only index supplied frames; no interpolation or ML is performed. Detail inspection of an imported ID still requires the API.

Event comparison retrieves two actual `/hotspots/{id}` details. No browser-generated classifications are included.

## Alerts

`GET /alerts?page=1&page_size=20&risk=<optional>` → `data: Alert[]` + pagination meta.

`Alert`: `id`, `title`, `description`, `created_at` strings; `type: "high_risk"|"abnormal"|"persistent"|"low_confidence"|"system"`; `risk: risk enum`; `hotspot_id: string|null`; `acknowledged: boolean`. Alerts with a hotspot ID navigate to the inspector. System notices need not have an associated hotspot. Acknowledgment mutation is outside current frontend scope.

## Reports

Requires `/health.capabilities.reports=true` for creating reports.
- `GET /reports?page=1&page_size=20` → `Report[]` with meta.
- `POST /reports` body `{ filters: FilterParams, format: "pdf" }` → `Report`. Validate and authorize on backend; avoid duplicate jobs using backend request/idempotency policy.
- `GET /reports/{id}` → `Report` for preview metadata/summary.
- `GET /reports/{id}/download` → PDF with `Content-Type: application/pdf` when ready. Authorize downloads server-side. Frontend uses blob downloads and never embeds executable report HTML.

`Report`: `id`, `name`, `created_at` strings; `status: "queued"|"processing"|"ready"|"failed"`; `summary: string|null`. List polls every 5 seconds only while a returned report is queued/processing. Preview uses actual backend summaries. Download is enabled only for ready reports. No report is generated locally.

## Analyst review

`POST /analyst-review` body `{ hotspot_id: string, classification: source enum, note: string, reviewer: string }` → `data: { classification, verifier, verified_at, note }`.

Requires `/health.capabilities.analyst_review=true`. The frontend requires a reviewer name and 10–2,000-character note; backend must validate again. **The reviewer field is descriptive, not authentication.** The backend MUST authorize the current principal, associate the audit identity server-side, reject spoofed verifier claims, and preserve the original AI prediction. UI does not mark a review verified before a valid success response and detail refetch. There is no local-only saved ground truth. Auth integration remains a backend/next-iteration task; this client currently sends requests without credentials.

## GIS layers and stream extension (not wired)

Industrial facilities and land-use map toggles are intentionally disabled in the current release. `gis_layers` is informational only; it must not imply current overlay rendering. Define and implement an authorized, bounds-limited GeoJSON/vector-tile endpoint before enabling those layers. Feature provenance must distinguish context from verification. Future stream events should reference changed entity IDs or invalidate existing query keys (`hotspots`, `summary`, `trends`, `alerts`, `historical`) rather than sending an unlimited database snapshot. No stream endpoint is claimed to exist.