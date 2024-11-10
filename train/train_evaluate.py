# train/train_evaluate.py

import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

from test import *
from .train_model import *
from compressor import *


def train_evaluate(npz_file):
    # Load the .npz dataset
    images, labels = decompress_npz(npz_file)

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

    val_dataset = ASLDataset(X_test, y_test, transform=transform)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Initialize the model
    model = CustomMobileNetV2(num_classes=num_classes)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Calculate class weights
    class_weights = compute_class_weight('balanced', classes=np.unique(labels), y=labels)
    class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)  # Send to device for GPU compatibility

    # Define loss with class weights and optimizer
    loss_fn = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    train_model(model, train_loader, val_loader, num_epochs, loss_fn, optimizer, device)

    evaluate_model(model, val_loader, device)


if __name__ == "__main__":
    train_evaluate('../data/compressed_asl_crop_v3.npz')
