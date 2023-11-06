    from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split

from leafy.loader import ImageLoader
from leafy.trainloader import ImageDataset

def initialise_dataset(folder):
    data_folder = Path("./images")
    loader = ImageLoader(data_folder=data_folder)
    image_db = ImageDataset(loader)
    train_set, test_set = random_split(image_db, [0.8, 0.2])

    # class_distribution = loader.get_better_class_distribution()
    im, y = image_db[0]
    num_classes = len(y)
    print(f"{im.shape = }\n{y.shape = }")