from pathlib import Path

from PIL import Image
from fire import Fire
from .loader import ImageLoader


from torchvision.transforms import RandomResizedCrop
from torchvision.transforms import RandomHorizontalFlip
from torchvision.transforms import RandomRotation
from torchvision.transforms import GaussianBlur
from torchvision.transforms import RandomAutocontrast
from torchvision.transforms import RandomAffine

from typing import Dict, Callable
from torch import Tensor

class Augmentinator:
    def __init__(self, augmentations: Dict[str, Callable[[Tensor], Tensor]]):
        self.augmentations = augmentations
    
    def augment(self, im: Tensor) -> Dict[str, Tensor]:
        return {name: aug(im) for name, aug in self.augmentations.items()}
        


# rc         = RandomResizedCrop(size= im.shape[1:], scale=(0.5, 1.0), ratio=(0.75, 1.3333333333333333), interpolation=2)
# flip       = RandomHorizontalFlip(p  =1.0)
# rot        = RandomRotation(degrees  =[0, 360])
# gauss      = GaussianBlur(kernel_size=5, sigma           =(0.1, 3.0))
# contraster = RandomAutocontrast(p    =1.0)
# shear      = RandomAffine(degrees    =[0,360], translate =(0.1, 0.3), scale=(0.9, 1.1), shear                        =(-30, 30))

# transformations = {
#     "crop"    : rc,
#     "flip"    : flip,
#     "rotation": rot,
#     "blur"    : gauss,
#     "contrast": contraster,
#     "shear"   : shear
# }

# aug = Augmentinator(transformations)
# augmented = aug.augment(im)
# augmented.keys()

# def augment_image(im: Image.Image) -> dict[str, Image.Image]:
