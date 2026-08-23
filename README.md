# SIH26162 — AI-Based Detection and Classification of Industrial Fires and Persistent Thermal Sources

Repository initialized according to the SIH26162 execution roadmap.

## Architecture

Ingestion → Enrichment → Inference → Presentation

- Python ML/data pipeline
- Express backend
- Firebase / Firestore
- React GIS dashboard

The primary classifier is tabular (XGBoost / LightGBM). Satellite imagery is an enhancement with a tabular-only fallback.

## Repository Structure

- `ml/` — ingestion, preprocessing, training, inference, models, datasets and imagery
- `server/` — Express backend
- `client/` — React GIS frontend
- `scripts/` — automation and validation scripts
- `docs/` — project documentation

## Core Classification

- `0` — Natural / vegetation fire
- `1` — Industrial fire

Persistent thermal source is derived separately from temporal behaviour.
