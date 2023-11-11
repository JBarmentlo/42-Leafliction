from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Module
from torch.utils.data import DataLoader, random_split
from torchvision.models import resnet50, ResNet50_Weights
from torch.optim import Adam
from tqdm import tqdm
import json
import shutil
from pathlib import Path

from .loader import ImageLoader
from .trainloader import ImageDataset
from .model import BasicClassifier


def initialise_dataset(folder):
    data_folder = Path(folder)
    loader = ImageLoader(data_folder=data_folder)
    image_db = ImageDataset(loader)
    return image_db


def train_model(data_loader, model, preprocess):
    ce_loss = nn.CrossEntropyLoss()
    model = model.cuda()
    optim = Adam(model.parameters())

    for _ in range(1):
        for x, y in tqdm(data_loader):
            optim.zero_grad()
            x = x.cuda()
            x = preprocess(x)
            y = y.cuda()
            y_hat = model(x)
            loss = ce_loss(y_hat, y)
            loss.backward()
            optim.step()
            # print(loss)

    return model


def eval_model(data_loader, model, preprocess):
    model.eval()
    gts = []
    preds = []

    for x, y in tqdm(data_loader):
        gts.append(torch.argmax(y, dim=1))
        y_hat = model(preprocess(x.cuda()))
        preds.append(torch.argmax(y_hat, dim=1).cpu())

    getes = torch.concat(gts)
    predes = torch.concat(preds)

    return predes, getes


def train(data_folder):
    image_db = initialise_dataset(data_folder)
    train_set, test_set = random_split(image_db, [0.8, 0.2])
    print(
        f"Split dataset into train: {len(train_set)} & validation: {len(test_set)}"
    )
    train_loader = DataLoader(train_set, 32, shuffle=True)
    model = BasicClassifier(len(image_db.class_labels))
    model = train_model(train_loader, model, model.preprocess)

    test_loader = DataLoader(test_set, 32, shuffle=True)

    preds, gts = eval_model(test_loader, model, model.preprocess)
    print(
        f"Accuracy on validation set: {(gts == preds).to(torch.float).mean() * 100:.0f}%"
    )

    shutil.rmtree("./model_save", ignore_errors=True)
    shutil.rmtree("./model_save.zip", ignore_errors=True)
    Path("./model_save").mkdir(exist_ok=True, parents=True)
    print("Saving model to ./model_save/model.pt")
    torch.save(model.state_dict(), "./model_save/model.pt")

    print("Dumping class labels to json.")
    with open("./model_save/classes.json", "w+") as f:
        json.dump({"classes": image_db.class_labels}, f)

    print("Copying images for some reason.")
    shutil.copytree(data_folder, "./model_save/images")

    print("Compressing archive.")
    shutil.make_archive("model_save", "zip", "./model_save")
    shutil.rmtree("./model_save", ignore_errors=True)
