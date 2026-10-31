"""Image helpers: composites and indices from multispectral arrays."""

import io

import numpy as np


def composite(x: np.ndarray, channel_names: list[str], bands: list[str]) -> np.ndarray:
    """(C,H,W) -> (H,W,3) from the named bands, with a 2–98 percentile stretch."""
    img = np.stack([x[channel_names.index(b)] for b in bands], -1)
    lo, hi = np.percentile(img, 2), np.percentile(img, 98)
    return np.clip((img - lo) / max(hi - lo, 1e-6), 0, 1)


def ndvi(x: np.ndarray, channel_names: list[str], roles: dict) -> np.ndarray:
    nir, red = x[channel_names.index(roles["nir"])], x[channel_names.index(roles["red"])]
    return (nir - red) / np.maximum(nir + red, 1e-6)


def views(channel_names: list[str], roles: dict) -> dict:
    """Which views are possible for this task: name -> bands (or 'ndvi')."""
    available = {}
    if all(r in roles for r in ("red", "green", "blue")):
        available["true_color"] = [roles["red"], roles["green"], roles["blue"]]
    else:
        available["first_3_channels"] = channel_names[:3]
    if all(r in roles for r in ("nir", "red", "green")):
        available["false_color"] = [roles["nir"], roles["red"], roles["green"]]
        available["ndvi"] = "ndvi"
    return available


def render(x: np.ndarray, channel_names: list[str], roles: dict, view: str, bands) -> tuple[np.ndarray, dict]:
    """Return (image, imshow kwargs) for one sample and one view."""
    if bands == "ndvi":
        return ndvi(x, channel_names, roles), {"cmap": "RdYlGn", "vmin": -0.2, "vmax": 0.9}
    return composite(x, channel_names, bands), {}


def png_bytes(fig) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=fig.dpi)
    return buf.getvalue()
