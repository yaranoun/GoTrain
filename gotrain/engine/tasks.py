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

# m-brick-kiln stores wrong band names too (geo-bench issue #29). True on-disk order, from the original
# export notebook and geo-bench's fixed converter, confirmed by band resolution + correlation fingerprints.
# B06, B09, B10 are absent; the last three channels are Sentinel-2's 8-bit true-colour image (TCI).
BRICK_KILN_DISK_ORDER = ["B01", "B02", "B03", "B04", "B05", "B07", "B8A", "B08", "B11", "B12",
                         "TCI_R", "TCI_G", "TCI_B"]

# m-cashew-plant: Sentinel-2 without B10, plus a cloud-probability layer. Its "B8A" is an exact
# copy of B08 in the source data (identical pixels, 10 m texture), kept as is.
CASHEW_DISK_ORDER = ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B11", "B12", "CLD"]


def _select(disk_order: list[str], names: list[str]) -> dict:
    return {"permutation": [disk_order.index(b) for b in names], "names": names}


# For each task: on-disk channels to keep (in this order), and their names.
CHANNELS = {
    "m-eurosat": _select(EUROSAT_DISK_ORDER, S2_CANONICAL),
    # TCI dropped: a rescaled duplicate of B04/B03/B02.
    "m-brick-kiln": _select(BRICK_KILN_DISK_ORDER, [b for b in S2_CANONICAL if b in BRICK_KILN_DISK_ORDER]),
    # Cloud probability dropped: 0 on >99% of pixels.
    "m-cashew-plant": _select(CASHEW_DISK_ORDER, CASHEW_DISK_ORDER[:12]),
    "m-pv4ger-seg": _select(["R", "G", "B"], ["R", "G", "B"]),
}

# Which channel plays which role, for composites and indices (true colour, false colour, NDVI).
S2_ROLES = {"red": "B04", "green": "B03", "blue": "B02", "nir": "B08", "swir1": "B11", "swir2": "B12"}
BAND_ROLES = {
    "m-eurosat": S2_ROLES,
    "m-brick-kiln": S2_ROLES,
    "m-cashew-plant": S2_ROLES,
    "m-pv4ger-seg": {"red": "R", "green": "G", "blue": "B"},
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
# Tasks not listed fall back to a per-image percentile stretch. 8-bit aerial RGB: 255.
DISPLAY_SCALE = {"m-eurosat": 3000.0, "m-brick-kiln": 3000.0, "m-cashew-plant": 3000.0, "m-pv4ger-seg": 255.0}


def display_scale(task: str) -> float | None:
    return DISPLAY_SCALE.get(task)


# m-eurosat also stores class names in the wrong order: the integer labels follow the
# original EuroSAT alphabetical order (verified visually: label 4 = industrial tanks,
# label 8 = rivers, label 9 = water). Labels themselves are correct; only names are fixed here.
CLASS_NAMES = {
    "m-eurosat": ["Annual Crop", "Forest", "Herbaceous Vegetation", "Highway", "Industrial Buildings",
                  "Pasture", "Permanent Crop", "Residential Buildings", "River", "Sea and Lake"],
    # m-cashew-plant ships without class names; these are the LABELS of geo-bench's converter.
    "m-cashew-plant": ["no data", "well-managed plantation", "poorly-managed plantation", "non-plantation",
                       "residential", "background", "uncertain"],
}

# Segmentation label excluded from loss, metrics and predictions. m-cashew-plant class 0 = "no data":
# pixels the label map doesn't cover (98% in fully unlabelled border rows/cols) while the image does,
# so it can't be learned from the imagery.
IGNORE_INDEX = {"m-cashew-plant": 0}


def ignore_index(task: str) -> int | None:
    return IGNORE_INDEX.get(task)


def info(task: str) -> dict:
    if task not in TASKS:
        raise ValueError(f"Unknown task '{task}'. Known: {sorted(TASKS)}")
    return TASKS[task]
