# model/CustomCNNs.py


import torch.nn as nn
from torchvision.models import (
    mobilenet_v2, MobileNet_V2_Weights,
    resnet50, ResNet50_Weights
)


class CustomMobileNetV2(nn.Module):
    def __init__(self, num_classes):
        super(CustomMobileNetV2, self).__init__()
        # Load the pre-trained MobileNetV2 model
        self.base_model = mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT)

        # Modify the classifier part of MobileNetV2
        self.base_model.classifier[1] = nn.Linear(self.base_model.last_channel, num_classes)

    def forward(self, x):
        return self.base_model(x)


class CustomResnet50(nn.Module):
    def __init__(self, num_classes):
        super(CustomResnet50, self).__init__()
        # Load the pre-trained ResNet50 model
        self.base_model = resnet50(weights=ResNet50_Weights.DEFAULT)

        # Modify the fully connected layer (classifier part) of ResNet50
        self.base_model.fc = nn.Linear(self.base_model.fc.in_features, num_classes)

    def forward(self, x):
        return self.base_model(x)
