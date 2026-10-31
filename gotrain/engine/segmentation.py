"""Segmentation pieces used by train.py / evaluate.py: loss, IoU metrics, prediction with TTA.

Masks are (N, H, W) integer class maps. D4 augmentation is applied to image and mask together;
TTA undoes each transform on the logits before averaging.
"""

import numpy as np
import torch
import torch.nn.functional as F

from gotrain.engine.augment import d4, d4_inverse


def logits(model, x: torch.Tensor, out_hw: tuple[int, int]) -> torch.Tensor:
    """Forward pass at any size: pad to a multiple of 32 (encoder stride), crop, resize to out_hw."""
    h, w = x.shape[-2:]
    ph, pw = (-h) % 32, (-w) % 32
    if ph or pw:
        x = F.pad(x, (0, pw, 0, ph), mode="reflect")
    out = model(x)[..., :h, :w]
    if tuple(out.shape[-2:]) != tuple(out_hw):  # img_size differs from the mask size
        out = F.interpolate(out.float(), size=out_hw, mode="bilinear", align_corners=False)
    return out


def loss_fn(out: torch.Tensor, y: torch.Tensor, label_smoothing: float = 0.0) -> torch.Tensor:
    """Cross-entropy + soft Dice (mean over classes present in the batch)."""
    out = out.float()
    ce = F.cross_entropy(out, y, label_smoothing=label_smoothing)
    probs = out.softmax(1)
    onehot = F.one_hot(y, out.shape[1]).permute(0, 3, 1, 2).float()
    inter = (probs * onehot).sum((0, 2, 3))
    denom = probs.sum((0, 2, 3)) + onehot.sum((0, 2, 3))
    present = onehot.sum((0, 2, 3)) > 0
    dice = (2 * inter + 1.0) / (denom + 1.0)
    return ce + (1 - dice[present]).mean()


def confusion(pred: torch.Tensor | np.ndarray, y: torch.Tensor | np.ndarray, n_classes: int) -> np.ndarray:
    """(n_classes, n_classes) pixel confusion matrix, rows = true class."""
    pred, y = torch.as_tensor(pred).reshape(-1).long(), torch.as_tensor(y).reshape(-1).long()
    cm = torch.bincount(y * n_classes + pred, minlength=n_classes * n_classes)
    return cm.reshape(n_classes, n_classes).cpu().numpy()


def iou_scores(cm: np.ndarray) -> dict:
    """Per-class IoU from a confusion matrix; mIoU averages classes that occur (in truth or prediction)."""
    tp = np.diag(cm).astype(np.float64)
    union = cm.sum(0) + cm.sum(1) - tp
    iou = np.where(union > 0, tp / np.maximum(union, 1), np.nan)
    return {
        "miou": float(np.nanmean(iou)),
        "per_class_iou": [None if np.isnan(v) else round(float(v), 4) for v in iou],
        "pixel_accuracy": float(tp.sum() / max(cm.sum(), 1)),
    }


@torch.no_grad()
def predict(model, x_all, prepare, device, tta=False, batch_size=16) -> np.ndarray:
    """Predicted class maps (N, H, W) uint8. tta=True averages softmax over all 8 D4 transforms."""
    model.eval()
    out = []
    for i in range(0, len(x_all), batch_size):
        xb = prepare(x_all[i:i + batch_size].to(device))
        hw = tuple(x_all.shape[-2:])
        with torch.autocast(device.type, enabled=device.type == "cuda"):
            if tta:
                p = torch.stack([d4_inverse(logits(model, d4(xb, k), hw).float().softmax(1), k)
                                 for k in range(8)]).mean(0)
            else:
                p = logits(model, xb, hw).float()
        out.append(p.argmax(1).to(torch.uint8).cpu())
    return torch.cat(out).numpy()


def per_image_iou(pred: np.ndarray, y: np.ndarray, n_classes: int) -> np.ndarray:
    """mIoU of each image (classes absent from both truth and prediction are skipped)."""
    return np.array([iou_scores(confusion(p, t, n_classes))["miou"] for p, t in zip(pred, y)])
