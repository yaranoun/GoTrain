"""Model zoo: backbones, pretrained weights, band adapters, classification/segmentation heads."""


def build_model(config: dict, n_channels: int, n_classes: int, task_type: str):
    # TODO: timm / torchgeo backbones; adapt first conv to n_channels; add head per task_type
    raise NotImplementedError
