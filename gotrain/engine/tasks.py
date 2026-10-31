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

RGB = {"m-eurosat": ["B04", "B03", "B02"]}


def info(task: str) -> dict:
    if task not in TASKS:
        raise ValueError(f"Unknown task '{task}'. Known: {sorted(TASKS)}")
    return TASKS[task]
