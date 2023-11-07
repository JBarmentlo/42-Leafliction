import torch
import json
from pathlib import Path
from PIL import Image
from torchvision.transforms.functional import to_tensor
import torch.nn.functional as F

from .model import BasicClassifier

def predict(image_path):
    model_folder = Path("./model_save")
    
    with (model_folder / "classes.json").open("r") as f:
        classes = json.load(f)["classes"]
    
    model = BasicClassifier(num_classes=len(classes))
    state_dict = torch.load(model_folder / "model.pt", map_location='cpu')
    model.load_state_dict(state_dict)
    
    im = model.preprocess(to_tensor(Image.open(image_path)).unsqueeze(0))
    
    pred = F.softmax(model(im), dim=1)
    print(pred)
    print(classes)
    print(classes[torch.argmax(pred).item()])
    
    