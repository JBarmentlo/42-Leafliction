import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch import Tensor
from torchvision.transforms.functional import to_tensor

from typing import List, Tuple



from .loader import ImageLoader

class ImageDataset(Dataset):
    def __init__(self, image_loader: ImageLoader):
        self.image_loader = image_loader
        self.images: Tensor
        self.ground_truth_str = []
        self.class_labels = self.compute_class_list()
        gt = Tensor(list(map(lambda label: self.class_labels.index(label), self.ground_truth_str)))
        self.gt = F.one_hot(gt.to(torch.int64), num_classes=len(self.class_labels)).to(torch.float32)
    
    def compute_class_list(self) -> List[str]:
        labels = set()
        images = []
        for im, label in self.image_loader:
            labels.add(label)
            self.ground_truth_str.append(label)
            images.append(im)
        
        self.images = torch.stack(images, dim=0)
        return list(labels)

    def __len__(self) -> int:
        return len(self.images)
    
    def __getitem__(self, index) -> Tuple[Tensor, Tensor]:
        return self.images[index], self.gt[index]