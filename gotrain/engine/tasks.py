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

# m-SA-crop-type: Sentinel-2 without B10, plus the converter's cloud layer (CLM). The stored names are right, checked
# on the HF v1.0 train split on 31 Oct (scripts/step9_inspect.py, 48 samples): B01/B09 show 6x6 blocks (60 m),
# B05/B06/B07/B8A/B11/B12 2x2 blocks (20 m), B02/B03/B04/B08 native 10 m texture; correlations match S2 (B07~B8A 0.99,
# B06~B07 0.98, B11~B12 0.97, B02~B03 0.96); the converter (crop_type_south_africa.py) lists the same order.
# VALUES ARE 8-BIT: 0-255 (int16 on disk, saturating at 255), about reflectance x 255, not reflectance x 1e4 like
# the other Sentinel-2 tasks (GEO-Bench's band_stats agree; true colour is natural at 76.5 = 0.3 reflectance).
# Cloud layer checked by hand: 0 on every pixel (the converter keeps each field tile's 5 least-cloudy dates), so it is
# dropped; a few images still show clouds. About 0.4-4% of pixels are 0 in every band (rotated-tile borders).
SA_CROP_DISK_ORDER = ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B11", "B12", "CLM"]


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
    # Cloud layer dropped: 0 everywhere.
    "m-SA-crop-type": _select(SA_CROP_DISK_ORDER, SA_CROP_DISK_ORDER[:12]),
    # Aerial 8-bit RGB (0.1 m), stored Red, Green, Blue; checked by a natural true-colour picture (31 Oct).
    "m-nz-cattle": _select(["R", "G", "B"], ["R", "G", "B"]),
}

# Which channel plays which role, for composites and indices (true colour, false colour, NDVI).
S2_ROLES = {"red": "B04", "green": "B03", "blue": "B02", "nir": "B08", "swir1": "B11", "swir2": "B12"}
BAND_ROLES = {
    "m-eurosat": S2_ROLES,
    "m-brick-kiln": S2_ROLES,
    "m-cashew-plant": S2_ROLES,
    "m-pv4ger-seg": {"red": "R", "green": "G", "blue": "B"},
    "m-SA-crop-type": S2_ROLES,
    "m-nz-cattle": {"red": "R", "green": "G", "blue": "B"},
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
# m-SA-crop-type is 8-bit (see SA_CROP_DISK_ORDER): 76.5 is the same 0.3 reflectance as 3000 elsewhere.
DISPLAY_SCALE = {"m-eurosat": 3000.0, "m-brick-kiln": 3000.0, "m-cashew-plant": 3000.0, "m-pv4ger-seg": 255.0,
                 "m-SA-crop-type": 3000.0, "m-nz-cattle": 255.0}

# Raw values multiplied (and rounded) when a split is loaded into the cache, so every Sentinel-2 task is in
# reflectance x 1e4 (what div10000 / SSL4EO weights expect). m-SA-crop-type is stored 8-bit (about reflectance x 255,
# see SA_CROP_DISK_ORDER): x 10000/255. Decided 31 Oct (build agent, on the second session's Step 9 finding); disclosed.
# Lossless: 256 distinct values map to 256 distinct integers.
VALUE_SCALE = {"m-SA-crop-type": 10000 / 255}


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
    # m-SA-crop-type ships without class names; these are crop_labels of geo-bench's converter (labels 0-9).
    "m-SA-crop-type": ["no data", "Lucerne/Medics", "Planted pastures (perennial)", "Fallow", "Wine grapes", "Weeds",
                       "Small grain grazing", "Wheat", "Canola", "Rooibos"],
    # m-nz-cattle: the task specs' names. Masks are radius-6 px boxes around point labels (nz_cattle_detection.py).
    "m-nz-cattle": ["no cattle", "cattle"],
}

# Segmentation label excluded from loss, metrics and predictions. m-cashew-plant class 0 = "no data":
# pixels the label map doesn't cover (98% in fully unlabelled border rows/cols) while the image does,
# so it can't be learned from the imagery.
# m-SA-crop-type class 0 = "No Data" (converter): unlabelled pixels, ignored the same way.
IGNORE_INDEX = {"m-cashew-plant": 0, "m-SA-crop-type": 0}


# Bootstrap groups for the noise gate and test intervals (CREATIVE_PLAN.md §4.2): samples that are not independent.
# m-cashew-plant has 25 dates per location; samples with an identical label mask are one location (val: 16 groups).
# m-SA-crop-type has 5 dates per field tile (names ..._<tile>_<k>, identical label masks, checked on train): one
# location per label mask (val: 250 groups of 1000 images).
# m-nz-cattle: about half its images have a "_shifted" twin of the same coordinates (train 204 pairs, val 6 pairs);
# nz_location groups a twin with its original by name (data.group_keys). 50 val locations also appear in train
# (GEO-Bench's own split): disclosed, not changed.
GROUP_BY = {"m-cashew-plant": "label_hash", "m-SA-crop-type": "label_hash", "m-nz-cattle": "nz_location"}


def ignore_index(task: str) -> int | None:
    return IGNORE_INDEX.get(task)


# Scope features for playbook lessons (CREATIVE_PLAN.md §4.5). A feature that cannot be derived is "unknown".
SENSOR_FAMILY = {  # sensor -> (sensor family, band family)
    "sentinel-2": ("s2-multispectral", "multispectral"),
    "aerial-rgb": ("aerial-rgb", "rgb"),
    "sentinel-1+2": ("s1+s2", "multispectral"),
}


def gsd_bucket(gsd_m: float | None) -> str:
    if gsd_m is None:
        return "unknown"
    return "<=1m" if gsd_m <= 1 else "(1,10]m" if gsd_m <= 10 else ">10m"


def scope(task: str) -> dict:
    """{task_type, sensor_family, band_family, gsd_bucket}; used to scope and retrieve lessons."""
    t = info(task)
    sensor, bands = SENSOR_FAMILY.get(t["sensor"], ("unknown", "unknown"))
    return {"task_type": t["type"] if t["type"] in ("classification", "segmentation") else "unknown",
            "sensor_family": sensor, "band_family": bands, "gsd_bucket": gsd_bucket(GSD_M.get(task))}


def info(task: str) -> dict:
    if task not in TASKS:
        raise ValueError(f"Unknown task '{task}'. Known: {sorted(TASKS)}")
    return TASKS[task]
