import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchvision.models as models
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split
from compressor import *
import numpy as np

# Load the .npz dataset
X, y = decompress_npz('C:/Users/xbpow/Downloads/Sign-Language-Recognition/data/compressed_asl.npz')

# ResNet18 expects 224x224 RGB images, we use transforms in PyTorch for this
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Pretrained ResNet normalization
])

# Apply the transformation to the dataset
X = np.array([transform(x) for x in X])

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert data to PyTorch tensors
X_train = torch.tensor(X_train)
X_test = torch.tensor(X_test)
y_train = torch.tensor(y_train)
y_test = torch.tensor(y_test)

# Create PyTorch datasets and dataloaders
train_dataset = data.TensorDataset(X_train, y_train)
test_dataset = data.TensorDataset(X_test, y_test)
train_loader = data.DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = data.DataLoader(test_dataset, batch_size=32, shuffle=False)

# Load pretrained ResNet18 model from torchvision with pretrained weights
resnet = models.resnet18(pretrained=True)

# Freeze the ResNet layers for transfer learning
for param in resnet.parameters():
    param.requires_grad = False

# Modify the final layer for your dataset
num_classes = len(np.unique(y_train))
resnet.fc = nn.Sequential(
    nn.Linear(resnet.fc.in_features, 1024),
    nn.ReLU(),
    nn.Linear(1024, num_classes)
)

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
resnet.to(device)

# Loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(resnet.fc.parameters(), lr=0.001)

# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    resnet.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = resnet(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    # Print the loss at the end of each epoch
    avg_loss = running_loss / len(train_loader)
    print(f'Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}')

# Save the model weights
torch.save(resnet.state_dict(), 'resnet18_custom.pth')
print("Model weights saved to 'resnet18_custom.pth'.")

# Evaluation
resnet.eval()
correct = 0
total = 0
with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = resnet(inputs)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total
print(f'Test Accuracy: {accuracy:.2f}%')
