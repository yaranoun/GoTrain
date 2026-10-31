"""Task registry: where each GEO-Bench task lives and how to read its channels correctly."""

# Canonical Sentinel-2 order, as expected by SSL4EO-S12 "ALL" weights (13 bands).
S2_CANONICAL = ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B10", "B11", "B12"]

TASKS = {
    # classification
    "m-eurosat": {"benchmark": "classification_v1.0", "type": "classification", "sensor": "sentinel-2"},
    "m-brick-kiln": {"benchmark": "classification_v1.0", "type": "classification", "sensor": "sentinel-2"},
    "m-so2sat": {"benchmark": "classification_v1.0", "type": "classification", "sensor": "sentinel-1+2"},
    "m-forestnet": {"benchmark": "classification_v1.0", "type": "classification", "sensor": "landsat-8"},
    "m-pv4ger": {"benchmark": "classification_v1.0", "type": "classification", "sensor": "aerial-rgb"},
    "m-bigearthnet": {"benchmark": "classification_v1.0", "type": "multilabel", "sensor": "sentinel-2"},
    # segmentation
    "m-pv4ger-seg": {"benchmark": "segmentation_v1.0", "type": "segmentation", "sensor": "aerial-rgb"},
    "m-chesapeake": {"benchmark": "segmentation_v1.0", "type": "segmentation", "sensor": "aerial-rgbn"},
    "m-cashew-plant": {"benchmark": "segmentation_v1.0", "type": "segmentation", "sensor": "sentinel-2"},
    "m-SA-crop-type": {"benchmark": "segmentation_v1.0", "type": "segmentation", "sensor": "sentinel-2"},
    "m-nz-cattle": {"benchmark": "segmentation_v1.0", "type": "segmentation", "sensor": "aerial-rgb"},
    "m-NeonTree": {"benchmark": "segmentation_v1.0", "type": "segmentation", "sensor": "aerial-rgb+hs+elev"},
}

# m-eurosat stores wrong band names (geo-bench issue #28). True on-disk channel order:
EUROSAT_DISK_ORDER = ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B09", "B10", "B11", "B12", "B8A"]

# For each task: permutation applied to on-disk channels, and the resulting channel names.
CHANNELS = {
    "m-eurosat": {
        "permutation": [EUROSAT_DISK_ORDER.index(b) for b in S2_CANONICAL],
        "names": S2_CANONICAL,
    },
}

# Which channel plays which role, for composites and indices (true colour, false colour, NDVI).
BAND_ROLES = {
    "m-eurosat": {"red": "B04", "green": "B03", "blue": "B02", "nir": "B08", "swir1": "B11", "swir2": "B12"},
}

GSD_M = {
    "m-eurosat": 10, "m-brick-kiln": 10, "m-so2sat": 10, "m-forestnet": 15, "m-pv4ger": 0.1,
    "m-bigearthnet": 10, "m-pv4ger-seg": 0.1, "m-chesapeake": 1, "m-cashew-plant": 10,
    "m-SA-crop-type": 10, "m-nz-cattle": 0.1, "m-NeonTree": 0.1,
}


def band_roles(task: str) -> dict:
    return BAND_ROLES.get(task, {})


# Sensor value shown at full brightness in composites (Sentinel-2 L1C reflectance * 10000).
# A fixed scale keeps the true R:G:B ratios; the per-image percentile stretch subtracts an
# offset that turns dark scenes (forest, water) blue from atmospheric haze in B02.
# Tasks not listed fall back to a per-image percentile stretch.
DISPLAY_SCALE = {"m-eurosat": 3000.0}


def display_scale(task: str) -> float | None:
    return DISPLAY_SCALE.get(task)

# m-eurosat also stores class names in the wrong order: the integer labels follow the
# original EuroSAT alphabetical order (verified visually: label 4 = industrial tanks,
# label 8 = rivers, label 9 = water). Labels themselves are correct; only names are fixed here.
CLASS_NAMES = {
    "m-eurosat": ["Annual Crop", "Forest", "Herbaceous Vegetation", "Highway", "Industrial Buildings",
                  "Pasture", "Permanent Crop", "Residential Buildings", "River", "Sea and Lake"],
}


def info(task: str) -> dict:
    if task not in TASKS:
        raise ValueError(f"Unknown task '{task}'. Known: {sorted(TASKS)}")
    return TASKS[task]
