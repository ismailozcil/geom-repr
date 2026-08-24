from typing import List, Sequence, Union
from PIL import Image
import torch
import torch.nn as nn
import torchvision.models as tvm
from geom_repr.datasets.transforms import build_standard_transform

BACKBONE_REGISTRY = {
    "resnet18": lambda: tvm.resnet18(weights=tvm.ResNet18_Weights.IMAGENET1K_V1), #[cite: 3]
    "resnet50": lambda: tvm.resnet50(weights=tvm.ResNet50_Weights.IMAGENET1K_V2), #[cite: 3]
    "resnet101": lambda: tvm.resnet101(weights=tvm.ResNet101_Weights.IMAGENET1K_V2), #[cite: 1]
    "resnet152": lambda: tvm.resnet152(weights=tvm.ResNet152_Weights.IMAGENET1K_V2), #[cite: 1]
    "resnext101": lambda: tvm.resnext101_32x8d(weights=tvm.ResNeXt101_32X8D_Weights.IMAGENET1K_V1), #[cite: 1]
    "regnet_y_16gf": lambda: tvm.regnet_y_16gf(weights=tvm.RegNet_Y_16GF_Weights.IMAGENET1K_V2), #[cite: 3]
    "efficientnet_v2_l": lambda: tvm.efficientnet_v2_l(weights=tvm.EfficientNet_V2_L_Weights.IMAGENET1K_V1), #[cite: 3]
    "vit_b_16": lambda: tvm.vit_b_16(weights=tvm.ViT_B_16_Weights.IMAGENET1K_V1), #[cite: 3]
    "vit_l_16": lambda: tvm.vit_l_16(weights=tvm.ViT_L_16_Weights.IMAGENET1K_V1), #[cite: 1]
    "dinov2_vits14": lambda: torch.hub.load("facebookresearch/dinov2", "dinov2_vits14"), #[cite: 2]
}


def get_backbone(model_name: str, device: torch.device = torch.device("cpu")) -> nn.Module: #[cite: 2]
    """Loads a pre-trained backbone, strips classification heads, and freezes weights[cite: 1, 2, 3]."""
    if model_name not in BACKBONE_REGISTRY:
        raise ValueError(f"Unknown backbone: '{model_name}'. Available: {list(BACKBONE_REGISTRY.keys())}")

    model = BACKBONE_REGISTRY[model_name]() #[cite: 3]
    if model_name.startswith("vit"): #[cite: 1, 3]
        model.heads = nn.Identity() #[cite: 1, 3]
    elif model_name.startswith("dinov2"):
        pass  # DINOv2 returns embedding directly[cite: 2]
    else:
        model = nn.Sequential(*list(model.children())[:-1]) #[cite: 2, 3]

    for p in model.parameters(): #[cite: 1, 2, 3]
        p.requires_grad_(False) #[cite: 1, 3]

    return model.to(device).eval() #[cite: 1, 2, 3]


@torch.no_grad()
def extract_features(
    images: Sequence[Union[str, Image.Image]],
    backbone: Union[str, nn.Module] = "resnet18",
    batch_size: int = 32,
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"), #[cite: 3]
    image_size: int = 224,
) -> torch.Tensor:
    """Extracts flattened feature representations for a list of image paths or PIL Images[cite: 3]."""
    if isinstance(backbone, str):
        extractor = get_backbone(backbone, device=device)
    else:
        extractor = backbone.to(device).eval()

    transform = build_standard_transform(size=image_size)
    all_feats: List[torch.Tensor] = []
    batch: List[torch.Tensor] = []

    for item in images:
        img = Image.open(item).convert("RGB") if isinstance(item, str) else item.convert("RGB") #[cite: 3]
        batch.append(transform(img)) #[cite: 3]
        if len(batch) == batch_size:
            x = torch.stack(batch).to(device) #[cite: 3]
            feats = extractor(x).flatten(1).cpu() #[cite: 3]
            all_feats.append(feats) #[cite: 3]
            batch = [] #[cite: 3]

    if batch:
        x = torch.stack(batch).to(device) #[cite: 3]
        feats = extractor(x).flatten(1).cpu() #[cite: 3]
        all_feats.append(feats) #[cite: 3]

    return torch.cat(all_feats, dim=0) if all_feats else torch.empty((0,)) #[cite: 3]
