# THERMOGRID

**Geospatial Thermal Intelligence for Smarter Fire Detection**

The analyst frontend for FireIntel AI. Built in the repository's original `client/` with React, TypeScript, Vite, MapLibre GL JS, TanStack Query, Zustand, Radix UI and Recharts. No ML pipeline code has been moved or rewritten.

## Run

Node 22+ is recommended (a transitive MapLibre dependency requires it). Yarn 1.22 is used with a committed lockfile. In the supplied Node 20.20 runtime, `yarn install --ignore-engines` is needed for that transitive engine declaration; type checking and the production build must still pass.

1. `cd client`
2. `yarn install --frozen-lockfile` (Node 22+)
3. Copy `.env.example` to `.env.local` and configure public values.
4. `yarn start`
5. `yarn lint` / `yarn typecheck` / `yarn build`
6. `yarn format` or `yarn format:check` for consistent source formatting.

The native compiler remains TypeScript 7 (`tsc`). ESLint's parser currently needs the TypeScript 6 programmatic API, so the official side-by-side aliases are installed: `@typescript/native` supplies TypeScript 7 and `typescript` resolves to the `@typescript/typescript6` compatibility API. This does not change the application compiler. ESLint 9 flat configuration is available at both the repository root and `client/`.

`PORT` and `HOST` configure the development server. The workspace runs on port 3000. A `/app/frontend` symlink to this existing client is solely a compatibility alias for the supplied supervisor; it is not a second frontend. All implementation belongs in `client/`.

The root README and `chat_context.md` describe an earlier Express / Firebase architecture. The current frontend request targets a future FastAPI / PostgreSQL / PostGIS API. The original Express skeleton is preserved, and **no REST detection endpoints are implemented by this frontend task**. The ML ingestion/inference files currently contain scheduled `NotImplementedError` stubs, not a working pipeline. They are unchanged.

## Data mode

API-only. `VITE_API_BASE_URL` is intentionally blank until a backend exists. No fabricated observations, mock adapter, silent fallback or simulated connection status is shipped. An unconfigured base URL produces a typed `NOT_CONFIGURED` error immediately without making a request. The real geographic basemap remains usable. A dash in KPI cards means unavailable, not zero observations.

All proposed routes, schemas, units and capability flags are in [`../docs/thermogrid-api-contract.md`](../docs/thermogrid-api-contract.md) and downloadable from Settings. These are **proposed contracts, not existing endpoints**. All JSON responses are runtime-validated with Zod and TypeScript models in `src/types/index.ts`. All data requests go through `src/services/`.

## Features and limitations

- Seven lazy-loaded routes: live map, dashboard, analytics, historical data, alerts, reports, settings; URL-based selected hotspot links.
- MapLibre geographic/satellite basemaps; real GeoJSON observations, clustering, hover, confidence opacity, risk halos, bounding queries, coordinate navigation, backend text search, geolocation, fullscreen and reset.
- Shared backend filters; pagination; bounded responses; canceled stale requests; cached API state; backend-confirmed polling.
- Separate environment and behaviour stages; model evidence; provenance; optical versus thermal imagery; review submission gated by backend capability.
- Historical frame replay and JSON import/export (20 MB file ceiling, 120 frames, 5,000 observations/frame). Imported files are explicitly labeled and not trusted as verified/live data.
- Reports are backend-generated; previews and downloads are gated by real report status.
- Industrial facility and land-use overlays remain visibly disabled until their backend GIS integration is implemented. They are not fabricated.
- Authentication is not implemented. The backend must authorize review/report actions and associate reviews with an authenticated principal; the entered reviewer name is not proof of identity.

## Data flow and state

Sources → backend ingestion/enrichment → PostgreSQL/PostGIS → Python two-stage inference → FastAPI → typed services → query cache → presentation.

`state/store.ts` separates ephemeral UI/filter/map state from persisted UI preferences. Selected IDs are URL state. TanStack Query owns remote API, analytics, historical and alert state. Polling stops when the backend is unavailable or capabilities do not confirm it; a stream adapter can later invalidate the same query keys. No probability inference is performed client-side; fraction-to-percentage formatting is presentation only.

## Maps and security

Basemap URLs and fonts are intentionally public client configuration. Esri Dark Gray Canvas base/reference and World Imagery are the current public raster providers. CARTO anonymous tiles were tested and replaced because they now show a key-required watermark. Comply with Esri/HERE/Garmin/OSM licensing and usage limits; attribution remains visible. Esri imagery is not a thermal feed and not necessarily recent. Public tiles can fail independently of the backend. Esri Dark Gray Canvas legacy services are scheduled for retirement in December 2029, so provider migration must be planned before then.

Never put FIRMS keys, Copernicus secrets, database URIs or server credentials in `VITE_*`. No private keys are requested by the frontend. API origin configuration is build-time, not writable from the UI. JSON contracts reject invalid coordinates/probabilities and non-HTTPS evidence images. Backend content is rendered as text, not HTML.

## Performance handoff

Map requests are extent-bounded and capped at 5,000 points; the server must enforce these limits, return `meta.truncated`, apply spatial indexes, and eventually provide vector tiles/server clusters for million-record workloads. The client never downloads an entire database. Table pages are 10/20 records. Chart series are capped at 366 points (90 requested); no chart fabricates continuity when an API fails.

## Testing

Production has no demo fixtures. Any synthetic contract responses used by browser tests are isolated in tests and must never become a runtime adapter. See `/app/test_reports/` for verification results when generated.