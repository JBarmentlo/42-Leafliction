import torch
from PIL import Image
from pathlib import Path
from torch.utils.data import DataLoader
from torchvision.transforms.functional import to_tensor, to_pil_image

from .data.Loader import ImageDataset
from .train import eval_model
from .models.basic import BasicClassifier
from .data.labels import LabelEnum
from .utils.image import double_im_with_text
from .data.Transformation import get_mask

def predict_file(image_paf: Path):
    device = "cpu"
    net = BasicClassifier()
    state_dict = torch.load("leaf_classifier.pt", map_location=device)
    net.load_state_dict(state_dict)
    net = net.eval()
    im = to_tensor(Image.open(str(image_paf)))
    x = im.unsqueeze(0)
    x = x.to(device)
    x = net.preprocess(x)
    y_hat = torch.argmax(torch.softmax(net(x), dim=1), dim=1)
    return LabelEnum(y_hat.item()).name


def evaluate_folder(data_folder: str):
    print("Evaluating: ", data_folder)
    folder = Path(data_folder)
    device = "cpu"
    net = BasicClassifier()
    state_dict = torch.load("leaf_classifier.pt", map_location=device)
    net.load_state_dict(state_dict)

    test_db = ImageDataset(folder)
    test_loader = DataLoader(test_db, batch_size=1, shuffle=False)

    print(eval_model(test_loader, net, device))


def predict(image_path):
    pred = predict_file(Path(image_path))
    im = Image.open(image_path)
    mask = get_mask
    im = to_tensor(Image.open(image_path))
    mask = get_mask(im)

    double_im_with_text(to_pil_image(im), to_pil_image(im * mask), pred).show()
