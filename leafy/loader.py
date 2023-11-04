from torch.utils.data import Dataset
from PIL import Image
from torchvision.transforms import ToTensor
from collections import Counter
from typing import Tuple
import torch
from pathlib import Path
from typing import Optional
from copy import deepcopy
from torch import Tensor

class ImageLoader(Dataset):
    def __init__(self, data_folder: Path, accept_globs: bool = True):
        self.data_folder  = data_folder
        self.accept_globs = accept_globs
        
        if self.accept_globs:
            self.image_files = list(Path.cwd().glob(str(self.data_folder / "**/*.JPG")))
        else:
            assert self.data_folder.is_dir()
            self.image_files = list(self.data_folder.glob("**/*.JPG"))
            
        self.totensor = ToTensor()
        self.fix_the_stupid_extension()
        print(f"Initiated loader on folder {data_folder.absolute()}. Found {len(self)} images.")
    
    def fix_the_stupid_extension(self):
        for im in self.image_files:
            if im.suffix == ".JPG":
                im.rename(im.with_suffix(".jpg"))
        
        if self.accept_globs:
            self.image_files = list(Path.cwd().glob(str(self.data_folder / "**/*.jpg")))
        else:
            assert self.data_folder.is_dir()
            self.image_files = list(self.data_folder.glob("**/*.jpg"))
    
    def get_sub_loader(self, desired_fruit: str, desired_disease: Optional[str] = None):
        new_image_paf_list = []
        for i, paf in enumerate(self.image_files):
            fruit, disease = self._get_fruit(paf), self._get_disease(paf)
            if fruit == desired_fruit:
                if desired_disease is None or disease == desired_disease:
                    new_image_paf_list.append(paf)
        
        sub_loader = deepcopy(self)
        sub_loader.image_files = new_image_paf_list
        return sub_loader
                
    @staticmethod
    def _get_fruit_and_disease(filename: str) -> Tuple[str, str]:
        fruit, *disease = filename.split("_")
        disease = '_'.join(disease)
        return fruit, disease
    
    @staticmethod
    def _get_fruit(path: Path) -> str:
        fruit, disease = ImageLoader._get_fruit_and_disease(path.parent.name)
        return fruit
    
    @staticmethod
    def _get_disease(path: Path) -> str:
        fruit, disease = ImageLoader._get_fruit_and_disease(path.parent.name)
        return disease
    
    def _get_class(self, path: Path) -> str:
        return path.parent.name

    def _get_image(self, path: Path) -> torch.Tensor:
        im = Image.open(path)
        return self.totensor(im)
    
    def get_class_distribution(self):
        class_lst = [im_class for _, im_class in self]
        return Counter(class_lst)
        
    def get_better_class_distribution(self):
        out = {}
        for k, v in self.get_class_distribution().items():
            fruit, disease = ImageLoader._get_fruit_and_disease(k)
            if fruit not in out:
                out[fruit] = {}
            out[fruit][disease] = v
        return out

    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, index) -> Tuple[Tensor, str]:
        return self._get_image(self.image_files[index]), self._get_class(self.image_files[index])
        
