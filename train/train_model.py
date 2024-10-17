# train/train_model.py

import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

from train import *
from test import *
from model import *
from compressor import *


def train_model(model, dataloader, num_epochs, loss_fn, optimizer, device):
    # Training loop
    for epoch in range(num_epochs):
        # Clear any cached GPU memory
        torch.cuda.empty_cache()

        running_loss = 0.0
        model.train()  # Set the model to training mode

        for images, labels in dataloader:
            # Move images and labels to the appropriate device (GPU or CPU)
            images, labels = images.to(device), labels.to(device)

            # Zero the gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(images)  # Forward pass with individual images

            # Calculate loss (ensure shapes are compatible)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        # Print loss for this epoch
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(dataloader):.4f}")

        # Save the model weights every few epochs (e.g., every 5 epochs)
        if (epoch + 1) % 5 == 0:
            torch.save(model.state_dict(), f'gesture_model_weights_epoch_{epoch + 1}.pth')
            print(f"Model weights saved for epoch {epoch + 1}.")

    print("Training finished.")


def train_evaluate():
    # Load the .npz dataset
    images, labels = decompress_npz('data/compressed_asl_crop.npz')

    # Split the images and labels into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize images
        transforms.ToTensor(),  # Convert to tensor
        transforms.Normalize(mean=mean, std=std)
    ])

    # Set up hyperparameters
    num_classes = len(np.unique(labels))
    num_epochs = 10
    batch_size = 100
    learning_rate = 0.001

    train_dataset = GestureDataset(X_train, y_train, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    test_dataset = GestureDataset(X_test, y_test, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Initialize the model
    model = CustomResnet18(num_classes=num_classes)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Define loss and optimizer
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    train_model(model, train_loader, num_epochs, loss_fn, optimizer, device)

    evaluate_model(model, test_loader, device)
