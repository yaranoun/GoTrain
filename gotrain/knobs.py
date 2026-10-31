"""Knob groups: what counts as "one change" between two configs (CREATIVE_PLAN.md §4.3), and recipe hashes.

Backbone, weights and what they derive (normalisation, image size, band fill) are one group: changing the weights
changes the expected input. Seeds, run caps and other bookkeeping keys are not knobs.
"""

import hashlib
import json

KNOB_GROUPS = {
    "model": ("backbone", "weights", "norm", "img_size", "band_fill"),
    "decoder": ("decoder",),
    "optimizer": ("optimizer", "lr", "weight_decay", "warmup_epochs"),
    "augment": ("augment",),
    "label_smoothing": ("label_smoothing",),
    "batch_size": ("batch_size",),
    "freeze_backbone": ("freeze_backbone",),
}
NOT_KNOBS = ("seed", "time_limit_min", "train_fraction", "epochs", "input_bands")
_GROUP_OF = {k: g for g, keys in KNOB_GROUPS.items() for k in keys}


def group_of(key: str) -> str | None:
    return _GROUP_OF.get(key)


def _canon(v):
    return json.dumps(v, sort_keys=True)


def group_values(cfg: dict, group: str) -> dict:
    return {k: cfg.get(k) for k in KNOB_GROUPS[group]}


def diff_groups(cfg_a: dict, cfg_b: dict) -> list[str]:
    """Knob groups whose values differ between two resolved configs (sorted)."""
    return sorted(g for g in KNOB_GROUPS if _canon(group_values(cfg_a, g)) != _canon(group_values(cfg_b, g)))


def recipe_hash(cfg: dict) -> str:
    """Hash of a resolved config without the non-knob keys: seed repeats share it."""
    recipe = {k: v for k, v in cfg.items() if k not in NOT_KNOBS and k != "task"}
    return "sha256:" + hashlib.sha256(_canon(recipe).encode()).hexdigest()
