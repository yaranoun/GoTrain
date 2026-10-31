"""Model zoo: backbones + pretrained weights, adapted to the task's channel count."""

import timm

from gotrain.engine.tasks import S2_CANONICAL

# What the agent will see from list_backbones(). "norm" is the input normalization the weights expect.
ZOO = {
    "resnet18/none": {"in_chans": "any", "norm": "zscore", "img_size": 64, "params_m": 11},
    "resnet18/imagenet": {"in_chans": "any", "norm": "zscore", "img_size": 64, "params_m": 11},
    "resnet50/imagenet": {"in_chans": "any", "norm": "zscore", "img_size": 64, "params_m": 23},
    "convnext_tiny/imagenet": {"in_chans": "any", "norm": "zscore", "img_size": 64, "params_m": 28},
    "resnet18/ssl4eo_moco": {"in_chans": 13, "bands": S2_CANONICAL, "norm": "div10000", "img_size": 224,
                             "params_m": 11, "notes": "Sentinel-2 L1C self-supervised (SSL4EO-S12 MoCo)"},
    "resnet50/ssl4eo_moco": {"in_chans": 13, "bands": S2_CANONICAL, "norm": "div10000", "img_size": 224,
                             "params_m": 23, "notes": "Sentinel-2 L1C self-supervised (SSL4EO-S12 MoCo)"},
}

# Segmentation: the backbone is the encoder of a decoder from segmentation_models_pytorch.
# img_size None = native resolution (fully convolutional; inputs padded to a multiple of 32).
SEG_ZOO = {
    "resnet18/imagenet": {"in_chans": "any", "norm": "zscore", "img_size": None, "params_m": 14},
    "resnet34/imagenet": {"in_chans": "any", "norm": "zscore", "img_size": None, "params_m": 24},
    "resnet50/imagenet": {"in_chans": "any", "norm": "zscore", "img_size": None, "params_m": 32},
    "resnet18/ssl4eo_moco": {"in_chans": 13, "bands": S2_CANONICAL, "norm": "div10000", "img_size": None,
                             "params_m": 14, "notes": "Sentinel-2 L1C self-supervised encoder (SSL4EO-S12 MoCo)"},
    "resnet50/ssl4eo_moco": {"in_chans": 13, "bands": S2_CANONICAL, "norm": "div10000", "img_size": None,
                             "params_m": 32, "notes": "Sentinel-2 L1C self-supervised encoder (SSL4EO-S12 MoCo)"},
}
DECODERS = {"unet": "Unet", "fpn": "FPN", "deeplabv3plus": "DeepLabV3Plus"}


def zoo_for(task_type: str) -> dict:
    return SEG_ZOO if task_type == "segmentation" else ZOO


def zoo_entry(task_type: str, backbone: str, weights: str) -> dict:
    zoo = zoo_for(task_type)
    key = f"{backbone}/{weights}"
    if key not in zoo:
        raise ValueError(f"Unknown backbone/weights '{key}' for {task_type}. Known: {sorted(zoo)}")
    return zoo[key]


def band_plan(task_type: str, backbone: str, weights: str, channel_names: list[str] | None) -> dict | None:
    """For weights pretrained on a fixed band list (SSL4EO: the 13 S2 bands): where each expected band comes
    from in the task's channels, and which are missing (filled in train.prepare). None = use channels as they are."""
    expected = zoo_entry(task_type, backbone, weights).get("bands")
    if not expected or (channel_names is not None and list(channel_names) == list(expected)):
        return None
    if channel_names is None:
        raise ValueError(f"{backbone}/{weights} needs named Sentinel-2 bands; this task's band names are unknown")
    other = [b for b in channel_names if b not in expected]
    if other:
        raise ValueError(f"{backbone}/{weights} needs Sentinel-2 bands; this task has other channels {other}")
    source = [channel_names.index(b) if b in channel_names else -1 for b in expected]
    return {"bands": list(expected), "source": source, "missing": [b for b, s in zip(expected, source) if s < 0]}


def build_model(backbone: str, weights: str, in_chans: int, n_classes: int,
                task_type: str = "classification", decoder: str = "unet"):
    entry = zoo_entry(task_type, backbone, weights)
    if entry["in_chans"] != "any" and entry["in_chans"] != in_chans:
        raise ValueError(f"{backbone}/{weights} needs exactly {entry['in_chans']} bands, task has {in_chans}")

    if task_type == "segmentation":
        return _build_segmenter(backbone, weights, in_chans, n_classes, decoder)

    if weights == "ssl4eo_moco":
        fn, w = _ssl4eo(backbone)
        return fn(weights=w.SENTINEL2_ALL_MOCO, num_classes=n_classes)

    # timm adapts the pretrained first conv to in_chans by repeating/scaling RGB filters.
    return timm.create_model(backbone, pretrained=(weights == "imagenet"), in_chans=in_chans, num_classes=n_classes)


def _ssl4eo(backbone: str):
    from torchgeo import models as tgm

    return {"resnet18": (tgm.resnet18, tgm.ResNet18_Weights),
            "resnet50": (tgm.resnet50, tgm.ResNet50_Weights)}[backbone]


def _build_segmenter(backbone, weights, in_chans, n_classes, decoder):
    import segmentation_models_pytorch as smp

    if decoder not in DECODERS:
        raise ValueError(f"Unknown decoder '{decoder}'. Known: {sorted(DECODERS)}")
    model = getattr(smp, DECODERS[decoder])(
        encoder_name=backbone,
        encoder_weights="imagenet" if weights == "imagenet" else None,  # smp adapts conv1 to in_chans
        in_channels=in_chans,
        classes=n_classes,
    )
    if weights == "ssl4eo_moco":
        # smp's ResNet encoder uses torchvision layer names, same as the timm ResNet torchgeo ships.
        _, w = _ssl4eo(backbone)
        state = {k: v for k, v in w.SENTINEL2_ALL_MOCO.get_state_dict(progress=False).items()
                 if not k.startswith("fc.")}
        expected = set(model.encoder.state_dict())
        missing, unexpected = sorted(expected - set(state)), sorted(set(state) - expected)
        if missing or unexpected:
            raise RuntimeError(f"SSL4EO weights don't fit the {backbone} encoder: "
                               f"missing={missing[:5]} unexpected={unexpected[:5]}")
        model.encoder.load_state_dict(state, strict=True)
    return model


def encoder_params(model, task_type: str):
    """Parameters frozen by freeze_backbone (everything except the head/decoder)."""
    if task_type == "segmentation":
        return list(model.encoder.parameters())
    head = set(id(p) for p in model.get_classifier().parameters())
    return [p for p in model.parameters() if id(p) not in head]
