# model/CustomCNNs.py


import torch.nn as nn
from torchvision.models import (
    mobilenet_v2, MobileNet_V2_Weights,
    resnet18, ResNet18_Weights
)


class CustomMobileNetV2(nn.Module):
    def __init__(self, num_classes):
        super(CustomMobileNetV2, self).__init__()
        # Load the pre-trained MobileNetV2 model
        self.base_model = mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT)

        # Change the classifier to output 1280 features instead of num_classes
        self.base_model.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(self.base_model.last_channel, 1280),  # Change output size to 1280
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1280, num_classes)  # Output layer for classification
        )

    def forward(self, x):
        return self.base_model(x)


class CustomResnet18(nn.Module):
    def __init__(self, num_classes):
        super(CustomResnet18, self).__init__()
        # Load the pre-trained ResNet50 model
        self.base_model = resnet18(weights=ResNet18_Weights.DEFAULT)

        # Modify the fully connected layer (classifier part) of ResNet50
        self.base_model.fc = nn.Sequential(
            nn.Linear(self.base_model.fc.in_features, 1280),  # Change output size to 1280
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1280, num_classes)  # Final output layer for classification
        )

    def forward(self, x):
        return self.base_model(x)
