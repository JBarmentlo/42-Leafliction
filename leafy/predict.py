import torch
import json
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from torchvision.transforms.functional import to_tensor, to_pil_image
import torch.nn.functional as F
import shutil
import sys

from .model import BasicClassifier
from .bullshit.Transformation import get_mask


def double_im_with_text(im1, im2, pred) -> Image.Image:
    w, h = im1.size
    margin = 10
    grid = Image.new("RGB", size=(2 * w + margin, 1 * h + 200))
    grid_w, grid_h = grid.size

    grid.paste(im1, box=(0, 0))
    grid.paste(im2, box=(w + margin, 0))

    draw = ImageDraw.Draw(grid)
    text = f"Predicted class: {pred}"
    fontsize = 20
    font = ImageFont.truetype("arial.ttf", fontsize)
    draw.text((10, h + 20), text, font=font)
    return grid


def predict(image_path):
    zip_archive = Path("./model_save.zip")

    if not zip_archive.exists():
        print("Train first please")
        sys.exit(0)

    shutil.unpack_archive(str(zip_archive), "./model_save")
    model_folder = Path("./model_save")

    with (model_folder / "classes.json").open("r") as f:
        classes = json.load(f)["classes"]

    model = BasicClassifier(num_classes=len(classes))
    state_dict = torch.load(model_folder / "model.pt", map_location="cpu")
    model.load_state_dict(state_dict)

    im = to_tensor(Image.open(image_path))
    mask = get_mask(im)

    pred = F.softmax(model(model.preprocess(im.unsqueeze(0))), dim=1)
    pred_class = classes[torch.argmax(pred).item()]
    double_im_with_text(
        to_pil_image(im), to_pil_image(im * mask), pred_class
    ).show()
