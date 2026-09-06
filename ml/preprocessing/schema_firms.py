"""FIRMS Parquet schema contract for Person B and Person A alignment.

Derived from SIH26162 Roadmap (Day 2 Core Fields):
- latitude (or lat)
- longitude (or lon)
- acq_date
- acq_time
- satellite
- instrument
- confidence
- bright_ti4
- bright_ti5
- frp
- daynight
- scan
- track
"""

FIRMS_SCHEMA = {
    "latitude": {
        "aliases": ["lat", "latitude"],
        "type": "float64",
        "description": "Center latitude of the fire pixel (WGS84, -90 to 90)",
        "required": True,
    },
    "longitude": {
        "aliases": ["lon", "longitude"],
        "type": "float64",
        "description": "Center longitude of the fire pixel (WGS84, -180 to 180)",
        "required": True,
    },
    "acq_date": {
        "aliases": ["acq_date"],
        "type": "string",
        "description": "Acquisition date in YYYY-MM-DD format",
        "required": True,
    },
    "acq_time": {
        "aliases": ["acq_time"],
        "type": "string",
        "description": "Acquisition time in HHMM format (UTC)",
        "required": True,
    },
    "satellite": {
        "aliases": ["satellite"],
        "type": "string",
        "description": "Satellite identification (e.g. N, Suomi-NPP, NOAA-20)",
        "required": True,
    },
    "instrument": {
        "aliases": ["instrument"],
        "type": "string",
        "description": "Instrument name (e.g. VIIRS, MODIS)",
        "required": True,
    },
    "confidence": {
        "aliases": ["confidence"],
        "type": "string",
        "description": "Detection confidence (e.g. l=low, n=nominal, h=high or 0-100)",
        "required": True,
    },
    "bright_ti4": {
        "aliases": ["bright_ti4"],
        "type": "float64",
        "description": "VIIRS I-4 thermal brightness temperature in Kelvin (375m)",
        "required": True,
    },
    "bright_ti5": {
        "aliases": ["bright_ti5"],
        "type": "float64",
        "description": "VIIRS I-5 thermal brightness temperature in Kelvin (375m)",
        "required": True,
    },
    "frp": {
        "aliases": ["frp"],
        "type": "float64",
        "description": "Fire Radiative Power in Megawatts (MW)",
        "required": True,
    },
    "daynight": {
        "aliases": ["daynight"],
        "type": "string",
        "description": "Day or Night observation flag ('D' or 'N')",
        "required": True,
    },
    "scan": {
        "aliases": ["scan"],
        "type": "float64",
        "description": "Pixel scan dimension in km",
        "required": True,
    },
    "track": {
        "aliases": ["track"],
        "type": "float64",
        "description": "Pixel track dimension in km",
        "required": True,
    },
}

CORE_FIELDS = list(FIRMS_SCHEMA.keys())


def validate_firms_columns(columns):
    """Validates that all required core fields (or acceptable aliases) exist in columns.

    Returns:
        tuple (bool, list): (is_valid, list_of_missing_fields)
    """
    col_set = set(columns)
    missing = []
    for field, spec in FIRMS_SCHEMA.items():
        if spec.get("required", False):
            if not any(alias in col_set for alias in spec["aliases"]):
                missing.append(field)
    return len(missing) == 0, missing
