import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50, ResNet50_Weights

class AWADNN(nn.Module):
    def __init__(self, classes = 50, model_type= 'res50', pretrained=True):
        super(AWADNN, self).__init__()
        if model_type == 'res50':
            self.model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        else:
            raise NotImplementedError()
        self.classes = classes
        self.model.fc = nn.Linear(self.model.fc.in_features, self.classes)
        
    def forward(self, x):
        x = self.model(x)
        return x