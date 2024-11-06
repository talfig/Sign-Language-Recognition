# model/cnn_models.py

import torch
import string
import torch.nn as nn
from torchsummary import summary
import torch.nn.functional as F
from torchvision.models import (
    mobilenet_v2, MobileNet_V2_Weights,
    resnet18, ResNet18_Weights
)


class CustomCNN(nn.Module):
    def __init__(self, num_classes):
        super(CustomCNN, self).__init__()

        # Convolutional layers with added batch normalization and dropout
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)

        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)

        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(128)

        self.conv4 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(256)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        # Dropout after pooling layers
        self.dropout_conv = nn.Dropout(0.2)

        # Fully connected layers with dropout
        self.fc1 = nn.Linear(256 * 14 * 14, 1024)  # Adjusted for additional conv layer
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, num_classes)
        self.dropout_fc = nn.Dropout(0.4)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.dropout_conv(x)

        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.dropout_conv(x)

        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = self.dropout_conv(x)

        x = self.pool(F.relu(self.bn4(self.conv4(x))))
        x = self.dropout_conv(x)

        x = x.view(-1, 256 * 14 * 14)  # Flatten for fully connected layers
        x = F.relu(self.fc1(x))
        x = self.dropout_fc(x)

        x = F.relu(self.fc2(x))
        x = self.dropout_fc(x)

        x = self.fc3(x)
        return x


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


class CustomResNet18(nn.Module):
    def __init__(self, num_classes):
        super(CustomResNet18, self).__init__()
        # Load the pre-trained ResNet50 model
        self.base_model = resnet18(weights=ResNet18_Weights.DEFAULT)

        # Modify the fully connected layer (classifier part) of ResNet18
        self.base_model.fc = nn.Sequential(
            nn.Linear(self.base_model.fc.in_features, 1280),  # Change output size to 1280
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1280, num_classes)  # Final output layer for classification
        )

    def forward(self, x):
        return self.base_model(x)


if __name__ == "__main__":
    # Total classes of uppercase letters (A-Z)
    total_classes = len(string.ascii_uppercase)
    model = CustomMobileNetV2(total_classes)

    # Move the models to the appropriate device (CPU or GPU)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    print("MobileNetV2 Summary:")
    summary(model, (3, 224, 224))
