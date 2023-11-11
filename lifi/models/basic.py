import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50, ResNet50_Weights
from icecream import ic
from torch.nn import Module

from ..data.labels import LabelEnum

class BasicClassifier(Module):
    def __init__(self, num_classes = len(LabelEnum)):
        super(BasicClassifier, self).__init__()
        resnet          = resnet50(weights=ResNet50_Weights.DEFAULT)
        modules         = list(resnet.children())[:-1]
        self.resnet     = nn.Sequential(*modules)
        self.fc1         = nn.Linear(2048, 256)
        self.fc2         = nn.Linear(256, num_classes)
        self.preprocess = ResNet50_Weights.DEFAULT.transforms()
        
    def forward(self, x):
        x = self.resnet(x)
        x = self.fc1(x.view(x.size(0), -1))
        x = self.fc2(x)
        return x
