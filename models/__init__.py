from .unet import UNet
from .unetpp import UNetPP
from .deeplabv3 import DeepLabV3Plus
from .ghost_unetpp import GRAUpp
from .convnext_upernet import ConvNeXtV2UperNet

MODEL_REGISTRY = {
    "unet": UNet,
    "unetpp": UNetPP,
    "deeplabv3": DeepLabV3Plus,
    "ghost_unetpp": GRAUpp,
    "convnext_upernet": ConvNeXtV2UperNet,
}


def build_model(name: str, input_channels: int = 12, num_classes: int = 5, **kwargs):
    """Instantiate one of the five benchmarked segmentation architectures."""
    key = name.lower()
    if key not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model '{name}'. Available: {list(MODEL_REGISTRY.keys())}")
    cls = MODEL_REGISTRY[key]
    if key in ("unet", "unetpp"):
        return cls(input_channels=input_channels, output_channels=num_classes, **kwargs)
    return cls(input_channels=input_channels, num_classes=num_classes, **kwargs)
