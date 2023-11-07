import sys
from pathlib import Path

from PIL import Image
from fire import Fire
from torchvision.transforms import RandomResizedCrop
from torchvision.transforms import RandomHorizontalFlip
from torchvision.transforms import RandomRotation
from torchvision.transforms import GaussianBlur
from torchvision.transforms import RandomAutocontrast
from torchvision.transforms import RandomAffine

from typing import Dict, Callable, Tuple
from torch import Tensor
from torchvision.transforms.functional import to_pil_image, to_tensor
import torch

from ..loader import ImageLoader
from .utils import image_grid

class Augmentinator:
    def __init__(self, augmentations: Dict[str, Callable[[Tensor], Tensor]]):
        self.augmentations = augmentations
    
    def _apply_augment(self, im: Tensor) -> Dict[str, Tensor]:
        return {name: aug(im) for name, aug in self.augmentations.items()}
    
    def make_image_row(self, ims: Dict[str, Tensor]):
        return to_pil_image(torch.cat([im for im in ims.values()], dim=2))

    def save_image_row(self, ims: Dict[str, Tensor], path: Path):
        for transform_name, im in ims.items():
            savepaf = (path.parent / (path.with_suffix("").name + '_' + transform_name)).with_suffix(path.suffix)
            to_pil_image(im).save(savepaf)
            
    
    def __call__(self, path: Path) -> Image.Image:
        if isinstance(path, str):
            path = Path(path)

        assert path.is_file(), f"Path {path} is not a file"
        assert path.suffix.lower() in [".jpg", ".png"], f"Path {path} is not a jpg or png file"
        
        im = to_tensor(Image.open(path))
        augmented =  self._apply_augment(im)
        self.save_image_row(augmented, path)
        return self.make_image_row(augmented)
        

def make_transforms(imshape: Tuple[int,int,int]) -> Dict[str, Callable[[Tensor], Tensor]]:
    rc         = RandomResizedCrop(size  =imshape[1:], scale=(0.5, 1.0), ratio=(0.75, 1.3333333333333333), interpolation=2)
    flip       = RandomHorizontalFlip(p  =1.0)
    rot        = RandomRotation(degrees  =[0, 360])
    gauss      = GaussianBlur(kernel_size=5, sigma           =(0.1, 3.0))
    contraster = RandomAutocontrast(p    =1.0)
    shear      = RandomAffine(degrees    =[0,360], translate =(0.1, 0.3), scale=(0.9, 1.1), shear                        =(-30, 30))

    transformations = {
        "crop"    : rc,
        "flip"    : flip,
        "rotation": rot,
        "blur"    : gauss,
        "contrast": contraster,
        "shear"   : shear
    }
    
    return transformations


def augment_single_image_from_path(paf: Path):
    im = to_tensor(Image.open(paf))

    transformations = make_transforms(im.shape)
    aug = Augmentinator(transformations)
    im = aug(paf)
    return im

def augmentation(*image_pafs):
    augs = []
    if len(image_pafs) == 0:
        print("Please provide a path as an argument")
        sys.exit(0)

    for image_paf in image_pafs:
        auged = augment_single_image_from_path(image_paf)
        augs.append(auged)
    image_grid(augs, rows=len(augs), cols=1).show()