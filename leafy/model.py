import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50, ResNet50_Weights
from icecream import ic
from torch.nn import Module


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
