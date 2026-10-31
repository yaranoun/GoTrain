"""Model zoo: backbones + pretrained weights, adapted to the task's channel count."""

import timm

# What the agent will see from list_backbones(). "norm" is the input normalization the weights expect.
ZOO = {
    "resnet18/none": {"in_chans": "any", "norm": "zscore", "img_size": 64, "params_m": 11},
    "resnet18/imagenet": {"in_chans": "any", "norm": "zscore", "img_size": 64, "params_m": 11},
    "resnet50/imagenet": {"in_chans": "any", "norm": "zscore", "img_size": 64, "params_m": 23},
    "convnext_tiny/imagenet": {"in_chans": "any", "norm": "zscore", "img_size": 64, "params_m": 28},
    "resnet18/ssl4eo_moco": {"in_chans": 13, "norm": "div10000", "img_size": 224, "params_m": 11,
                             "notes": "Sentinel-2 self-supervised; needs all 13 bands in canonical order"},
    "resnet50/ssl4eo_moco": {"in_chans": 13, "norm": "div10000", "img_size": 224, "params_m": 23,
                             "notes": "Sentinel-2 self-supervised; needs all 13 bands in canonical order"},
}


def build_model(backbone: str, weights: str, in_chans: int, n_classes: int):
    key = f"{backbone}/{weights}"
    if key not in ZOO:
        raise ValueError(f"Unknown backbone/weights '{key}'. Known: {sorted(ZOO)}")

    if weights == "ssl4eo_moco":
        if in_chans != 13:
            raise ValueError("ssl4eo_moco weights need exactly 13 Sentinel-2 bands")
        from torchgeo import models as tgm

        builder = {"resnet18": (tgm.resnet18, tgm.ResNet18_Weights), "resnet50": (tgm.resnet50, tgm.ResNet50_Weights)}
        fn, w = builder[backbone]
        return fn(weights=w.SENTINEL2_ALL_MOCO, num_classes=n_classes)

    # timm adapts the pretrained first conv to in_chans by repeating/scaling RGB filters.
    return timm.create_model(backbone, pretrained=(weights == "imagenet"), in_chans=in_chans, num_classes=n_classes)
