"""Image helpers: composites and indices from multispectral arrays."""

import io

import numpy as np


def composite(x: np.ndarray, channel_names: list[str], bands: list[str],
              scale: float | None = None, gamma: float = 1.4) -> np.ndarray:
    """(C,H,W) -> (H,W,3) from the named bands; one stretch shared by all three channels.

    scale: sensor value shown at full brightness (see tasks.display_scale), with a gamma lift.
    None: per-image 2–98 percentile stretch (for sensors without a known scale).
    """
    img = np.stack([x[channel_names.index(b)] for b in bands], -1).astype(np.float32)
    if scale is not None:
        return np.clip(img / scale, 0, 1) ** (1 / gamma)
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


def render(x: np.ndarray, channel_names: list[str], roles: dict, view: str, bands,
           scale: float | None = None) -> tuple[np.ndarray, dict]:
    """Return (image, imshow kwargs) for one sample and one view."""
    if bands == "ndvi":
        return ndvi(x, channel_names, roles), {"cmap": "RdYlGn", "vmin": -0.2, "vmax": 0.9}
    return composite(x, channel_names, bands, scale=scale), {}


def png_bytes(fig) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=fig.dpi)
    return buf.getvalue()
