"""Explore tools: what the agent uses to get to know its dataset before planning (bound to one session)."""

from pathlib import Path

from strands import tool

from gotrain.engine import profile as prof
from gotrain.session import Session
from gotrain.tools._result import EXPECTED, error


def make_tools(session: Session) -> dict:
    task = session.task

    @tool
    def profile_dataset() -> dict:
        """Profile this dataset before planning: band statistics, class balance, and sample image grids.

        Returns a text summary plus images to look at: true colour (or the first 3 channels), false colour
        (NIR-red-green) and NDVI when the band roles are known. For segmentation, the label mask is drawn over
        each image. Only train and val are used; test stays locked.
        """
        try:
            p = prof.profile(task)
        except EXPECTED as e:
            return error(str(e))
        content = [{"text": prof.summary_text(p)}]
        for path in p["images"]:
            content.append({"text": f"Image: {Path(path).stem}"})
            content.append({"image": {"format": "png", "source": {"bytes": Path(path).read_bytes()}}})
        content.append({"text": "Record what you notice with save_profile_notes before planning."})
        return {"status": "success", "content": content}

    @tool
    def save_profile_notes(notes: str) -> str:
        """Save your observations about this dataset (what the images and stats suggest for modelling).

        Args:
            notes: short observations, e.g. which classes look alike, clouds, useful bands.
        """
        prof.save_notes(task, notes)
        return "Notes saved to the dataset profile."

    return [profile_dataset, save_profile_notes]
