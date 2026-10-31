"""Strands tool results: text + optional PNG images, or an error the agent can read and act on."""

from pathlib import Path


def ok(text: str, images: list[tuple[str, bytes | str | Path]] = ()) -> dict:
    content = [{"text": text}]
    for title, img in images:
        data = Path(img).read_bytes() if isinstance(img, (str, Path)) else img
        content.append({"text": f"Image: {title}"})
        content.append({"image": {"format": "png", "source": {"bytes": data}}})
    return {"status": "success", "content": content}


def error(message: str) -> dict:
    return {"status": "error", "content": [{"text": message}]}


EXPECTED = (ValueError, PermissionError, FileNotFoundError, RuntimeError)
