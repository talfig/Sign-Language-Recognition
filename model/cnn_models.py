# model/cnn_models.py

import torch
import string
from torchsummary import summary
import torch.nn.functional as F
from .attention_layers import *
from torchvision.models import (
    mobilenet_v2, MobileNet_V2_Weights,
    resnet18, ResNet18_Weights
)


class CustomCNN(nn.Module):
    def __init__(self, num_classes):
        super(CustomCNN, self).__init__()

        # Convolutional layers with batch normalization and dropout
        self.conv1 = nn.Conv2d(3, 32, 3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.attention1 = ChannelAttention(32)  # Attention after conv1

        self.conv2 = nn.Conv2d(32, 64, 3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.attention2 = ChannelAttention(64)  # Attention after conv2

        self.conv3 = nn.Conv2d(64, 128, 3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.attention3 = ChannelAttention(128)  # Attention after conv3

        self.conv4 = nn.Conv2d(128, 256, 3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.attention4 = ChannelAttention(256)  # Attention after conv4

        self.pool = nn.MaxPool2d(2, 2)
        self.dropout_conv = nn.Dropout(0.2)
        self.fc1 = nn.Linear(256 * 14 * 14, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, num_classes)
        self.dropout_fc = nn.Dropout(0.4)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.attention1(x)
        x = self.dropout_conv(x)

        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.attention2(x)
        x = self.dropout_conv(x)

        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = self.attention3(x)
        x = self.dropout_conv(x)

        x = self.pool(F.relu(self.bn4(self.conv4(x))))
        x = self.attention4(x)
        x = self.dropout_conv(x)

        x = x.view(-1, 256 * 14 * 14)
        x = F.relu(self.fc1(x))
        x = self.dropout_fc(x)
        x = F.relu(self.fc2(x))
        x = self.dropout_fc(x)
        x = self.fc3(x)
        return x


class CustomMobileNetV2(nn.Module):
    def __init__(self, num_classes):
        super(CustomMobileNetV2, self).__init__()
        self.base_model = mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT)

        # Add channel attention after key layers
        self.attention1 = ChannelAttention(32)
        self.attention2 = ChannelAttention(96)
        self.attention3 = ChannelAttention(1280)

        self.base_model.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(self.base_model.last_channel, 1280),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1280, num_classes)
        )

    def forward(self, x):
        x = self.base_model.features[0](x)
        x = self.attention1(x)

        for i in range(1, 6):
            x = self.base_model.features[i](x)
        x = self.attention2(x)

        for i in range(6, len(self.base_model.features)):
            x = self.base_model.features[i](x)
        x = self.attention3(x)

        x = x.mean([2, 3])
        x = self.base_model.classifier(x)
        return x


class CustomResNet18(nn.Module):
    def __init__(self, num_classes):
        super(CustomResNet18, self).__init__()
        self.base_model = resnet18(weights=ResNet18_Weights.DEFAULT)

        # Add channel attention after each stage
        self.attention1 = ChannelAttention(64)
        self.attention2 = ChannelAttention(128)
        self.attention3 = ChannelAttention(256)
        self.attention4 = ChannelAttention(512)

        self.base_model.fc = nn.Sequential(
            nn.Linear(self.base_model.fc.in_features, 1280),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1280, num_classes)
        )

    def forward(self, x):
        x = self.base_model.conv1(x)
        x = self.base_model.bn1(x)
        x = self.base_model.relu(x)
        x = self.base_model.maxpool(x)
        x = self.attention1(x)

        x = self.base_model.layer1(x)
        x = self.attention2(x)

        x = self.base_model.layer2(x)
        x = self.attention3(x)

        x = self.base_model.layer3(x)
        x = self.attention4(x)

        x = self.base_model.layer4(x)
        x = self.base_model.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.base_model.fc(x)
        return x


if __name__ == "__main__":
    # Total classes of uppercase letters (A-Z)
    total_classes = len(string.ascii_uppercase)
    model = CustomMobileNetV2(total_classes)

    # Move the models to the appropriate device (CPU or GPU)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    print("MobileNetV2 Summary:")
    summary(model, (3, 224, 224))
