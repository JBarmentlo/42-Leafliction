from torch.utils.data import Dataset
from PIL import Image
from torchvision.transforms import ToTensor
from collections import Counter
from typing import Tuple
import torch
from pathlib import Path

class ImageLoader(Dataset):
    def __init__(self, data_folder: Path):
        self.data_folder = data_folder
        self.image_files = list(self.data_folder.glob("**/*.JPG"))
        self.totensor = ToTensor()
        assert self.data_folder.is_dir()
    
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
    
    def __getitem__(self, index):
        return self._get_image(self.image_files[index]), self._get_class(self.image_files[index])
        
