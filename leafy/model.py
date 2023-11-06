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
        self.fc = nn.Linear(2048, 8)
        
    def forward(self, x):
        x = self.resnet(x)
        x = self.fc(x.view(x.size(0), -1))
        return x
        # return F.softmax(x, dim=1) # will break for unbatched data


net = BasicClassifier(num_classes = 8)
net = net.cuda()


preprocess = ResNet50_Weights.DEFAULT.transforms()
# resnet = resnet50(weights=ResNet50_Weights.DEFAULT)
