from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Module
from torch.utils.data import DataLoader, random_split
from torchvision.models import resnet50, ResNet50_Weights
from torch.optim import Adam
from tqdm import tqdm

from .loader import ImageLoader
from .trainloader import ImageDataset

class BasicClassifier(Module):
    def __init__(self, num_classes):
        super(BasicClassifier, self).__init__()
        resnet = resnet50(weights=ResNet50_Weights.DEFAULT)
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.fc = nn.Linear(2048, num_classes)
        self.preprocess = ResNet50_Weights.DEFAULT.transforms()
        
    def forward(self, x):
        x = self.resnet(x)
        x = self.fc(x.view(x.size(0), -1))
        return x


def initialise_dataset(folder):
    data_folder = Path(folder)
    loader      = ImageLoader(data_folder=data_folder)
    image_db    = ImageDataset(loader)
    return image_db

def train_model(data_loader, model, preprocess):
    ce_loss = nn.CrossEntropyLoss()
    model = model.cuda()
    optim = Adam(model.parameters())

    for epoch in range(1):
        for x, y in tqdm(data_loader):
            optim.zero_grad()
            x = x.cuda()
            x = preprocess(x)
            y = y.cuda()
            y_hat = model(x)
            loss = ce_loss(y_hat, y)
            loss.backward()
            optim.step()
            print(loss)

    return model

def eval_model(data_loader, model, preprocess):
    model.eval()
    gts = []
    preds = []

    for x, y in tqdm(data_loader):
        gts.append(torch.argmax(y, dim=1))
        y_hat = model(preprocess(x.cuda()))
        preds.append(torch.argmax(y_hat, dim=1).cpu())
    
    getes  = torch.concat(gts)
    predes = torch.concat(preds)
    
    return predes, getes


def train(data_folder):
    image_db = initialise_dataset(data_folder)
    train_set, test_set = random_split(image_db, [0.8, 0.2])
    train_loader = DataLoader(train_set, 32, shuffle = True)
    model = BasicClassifier(len(image_db.class_labels))
    model = train_model(train_loader, model, model.preprocess)
    
    test_loader = DataLoader(test_set, 32, shuffle = True)
    
    
    preds, gts = eval_model(test_loader, model, model.preprocess)
    print(f"PERF: {(gts == preds).to(torch.float).mean()}")