from pathlib import Path
from collections import Counter
import torch
from torch.optim import Adam
from torch.utils.data import WeightedRandomSampler, random_split, DataLoader
from torchvision.transforms import RandomApply
from tqdm import tqdm
from sklearn.metrics import classification_report
import pandas as pd
import numpy as np

from .data.Loader import ImageDataset, LabelEnum
from .data.Augmentinator import default_transforms
from .models.basic import BasicClassifier


def get_label_probabilities(db: ImageDataset):
    gt = [y for x, y in db]
    count = Counter(gt)
    total = sum(count.values())
    p = {label: float(n) / total for label, n in count.items()}
    label_p = list(map(lambda x: p[x], gt))
    return label_p


def prepare_data(db: ImageDataset):
    torch.manual_seed(42)
    train_db, test_db = random_split(db, [0.7, 0.3])
    label_probas = get_label_probabilities(train_db)
    sampler = WeightedRandomSampler(label_probas, len(train_db))
    train_loader = DataLoader(train_db, 32, num_workers=4, sampler=sampler)
    test_loader = DataLoader(test_db, 32, num_workers=4)
    return train_loader, test_loader


def get_transforms(db):
    x, y = db[0]
    transform_dict = default_transforms(x.shape)
    transforms = RandomApply(
        torch.nn.ModuleList([t for t in transform_dict.values()]), p=0.1
    )
    return transforms


def eval_model(
    test_loader: DataLoader,
    net: torch.nn.Module,
    device="cuda" if torch.cuda.is_available() else "cpu",
):
    preds, gts = [], []

    net = net.to(device)
    net = net.eval()

    for x, y in tqdm(test_loader, desc="Evaluating model"):
        x = x.to(device)
        x = net.preprocess(x)
        y = y.to(device)
        y_hat = torch.argmax(torch.softmax(net(x), dim=1), dim=1)
        preds.append(y_hat.cpu())
        gts.append(y.cpu())

    net = net.train()

    preds_np = np.array(torch.concat(preds).view(-1))
    gts_np = np.array(torch.concat(gts).view(-1))

    report = classification_report(
        gts_np,
        preds_np,
        target_names=[LabelEnum(i).name for i in range(len(LabelEnum))],
        output_dict=True,
    )
    return pd.DataFrame.from_dict(report)


def train(
    data_folder: Path,
    epochs=1,
    device="cuda" if torch.cuda.is_available() else "cpu",
):
    data_folder = Path("./images")
    db = ImageDataset(data_folder)
    train_loader, test_loader = prepare_data(db)
    transforms = get_transforms(db)

    net = BasicClassifier(num_classes=len(LabelEnum))

    ce_loss = torch.nn.CrossEntropyLoss()
    net = net.to(device)
    optim = Adam(net.parameters())

    for e in range(epochs):
        print(f"Epoch: {e:2d}/{epochs:2d}")
        for x, y in tqdm(train_loader, desc="Training model"):
            optim.zero_grad()
            x = x.to(device)
            x = net.preprocess(x)
            x = transforms(x)
            y = y.to(device)
            y_hat = net(x)
            loss = ce_loss(y_hat, y)
            loss.backward()
            optim.step()

        report = eval_model(test_loader, net, device)
        print(report)

    torch.save(net.state_dict(), "leaf_classifier.pt")
