from torch.utils.data import Dataset
from PIL import Image
from torchvision.transforms import ToTensor
from pathlib import Path
import torch
from typing import Tuple
from collections import Counter

class ImageDataset(Dataset):
    def __init__(self, data_folder: Path):
        self.data_folder = data_folder
        self.image_files = list(self.data_folder.glob("**/*.jpg"))
        self.totensor    = ToTensor()

    def get_better_class_distribution(self):
        out = {}
        class_lst = [im_class for _, im_class in self]
        class_counter =  Counter(class_lst)
        for k, v in class_counter.items():
            fruit, disease = self._get_fruit_and_disease(k)
            if fruit not in out:
                out[fruit] = {}
            out[fruit][disease] = v
        return out

    def _get_fruit_and_disease(self, filename: str) -> Tuple[str, str]:
        fruit, *disease = filename.split("_")
        disease = '_'.join(disease)
        return fruit, disease
    
    def _get_class(self, path: Path) -> str:
        return path.parent.name

    def _get_image(self, path: Path) -> torch.Tensor:
        im = Image.open(path)
        return self.totensor(im)

    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, index) -> Tuple[torch.Tensor, str]:
        return self._get_image(self.image_files[index]), self._get_class(self.image_files[index])
        