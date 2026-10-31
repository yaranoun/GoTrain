"""Explore tools: what the agent uses to get to know a dataset before planning."""

from pathlib import Path

from strands import tool

from gotrain.engine import profile as prof
from gotrain.engine.tasks import TASKS


@tool
def list_tasks() -> str:
    """List the GEO-Bench tasks GoTrain knows about, with task type and sensor."""
    return "\n".join(f"{name}: {t['type']}, {t['sensor']}" for name, t in TASKS.items())


@tool
def profile_dataset(task: str) -> dict:
    """Profile a dataset before planning: band statistics, class balance, and sample image grids.

    Returns a text summary plus images to look at: true colour (or the first 3 channels),
    false colour (NIR-red-green) and NDVI when the band roles are known. For segmentation,
    the label mask is drawn over each image. Only train and val are used; test stays locked.

    Args:
        task: GEO-Bench task name, e.g. "m-eurosat".
    """
    p = prof.profile(task)
    content = [{"text": prof.summary_text(p)}]
    for path in p["images"]:
        content.append({"text": f"Image: {Path(path).stem}"})
        content.append({"image": {"format": "png", "source": {"bytes": Path(path).read_bytes()}}})
    content.append({"text": "Record what you notice with save_profile_notes before planning."})
    return {"status": "success", "content": content}


@tool
def save_profile_notes(task: str, notes: str) -> str:
    """Save your observations about a dataset (what the images and stats suggest for modelling).

    Args:
        task: GEO-Bench task name.
        notes: Short observations, e.g. which classes look alike, clouds, useful bands.
    """
    prof.save_notes(task, notes)
    return f"Notes saved to the {task} profile."
