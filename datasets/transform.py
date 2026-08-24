import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image
from typing import Union


class SquarePad:
    """Pads image or tensor to square while maintaining aspect ratio[cite: 2, 3]."""

    def __call__(self, img: Union[Image.Image, torch.Tensor]) -> torch.Tensor: #[cite: 2, 3]
        if isinstance(img, Image.Image):
            w, h = img.size
            max_side = max(h, w)
            pad_left = (max_side - w) // 2
            pad_top = (max_side - h) // 2
            pad_right = max_side - w - pad_left
            pad_bottom = max_side - h - pad_top
            return T.functional.pad(img, (pad_left, pad_top, pad_right, pad_bottom), fill=0)
        elif isinstance(img, torch.Tensor): #[cite: 2]
            *_, h, w = img.shape #[cite: 3]
            max_side = max(h, w) #[cite: 2, 3]
            pad_h = max_side - h #[cite: 2, 3]
            pad_w = max_side - w #[cite: 2, 3]
            padding = (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2) #[cite: 2, 3]
            return F.pad(img, padding, mode="constant", value=0.0) #[cite: 2, 3]
        else:
            raise TypeError("Input must be a PIL Image or PyTorch Tensor.") #[cite: 2]


def build_standard_transform(size: int = 224) -> T.Compose:
    """Standard pre-processing pipeline for ImageNet-pretrained backbones[cite: 1, 2, 3]."""
    return T.Compose([ #[cite: 1, 2, 3]
        T.ToTensor(), #[cite: 1, 2, 3]
        SquarePad(), #[cite: 2, 3]
        T.Resize(size, antialias=True), #[cite: 1, 2, 3]
        T.CenterCrop(size), #[cite: 1, 2, 3]
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), #[cite: 1, 2, 3]
    ])
