# test/train_test.py

import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

from test.evaluate_model import *
from train import *
from compressor import *


def train_evaluate():
    # Load the .npz dataset
    images, labels = decompress_npz('../data/compressed_asl_mediapipe.npz')

    # Split the images and labels into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),  # Resize images
        transforms.ToTensor(),  # Convert to tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Set up hyperparameters
    num_classes = len(np.unique(labels))
    num_epochs = 10
    batch_size = 100
    learning_rate = 0.001

    train_dataset = ASLDataset(X_train, y_train, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    test_dataset = ASLDataset(X_test, y_test, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Initialize the model
    model = CustomMobileNetV2(num_classes=num_classes)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Define loss and optimizer
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    train_model(model, train_loader, num_epochs, loss_fn, optimizer, device)

    evaluate_model(model, test_loader, device)


if __name__ == "__main__":
    train_evaluate()
