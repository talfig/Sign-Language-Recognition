# train/train_model.py

from utils import *


def run_epoch(model, dataloader, loss_fn, optimizer, device, train=True):
    """Runs a single epoch for training or validation and returns the average loss and accuracy."""
    if train:
        model.train()  # Set model to training mode
    else:
        model.eval()  # Set model to evaluation mode

    running_loss = 0.0
    correct_preds = 0
    total_preds = 0

    with torch.set_grad_enabled(train):  # Enable gradients only if training
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)

            # Zero gradients only if training
            if train:
                optimizer.zero_grad()

            # Forward pass
            outputs = model(images)
            loss = loss_fn(outputs, labels)
            running_loss += loss.item()

            # Backward pass and optimization only if training
            if train:
                loss.backward()
                optimizer.step()

            # Calculate accuracy
            _, preds = torch.max(outputs, 1)
            correct_preds += torch.eq(preds, labels).float().sum()
            total_preds += labels.size(0)

    avg_loss = running_loss / len(dataloader)
    avg_accuracy = (correct_preds / total_preds).item()

    return avg_loss, avg_accuracy


def train_model(model, train_loader, val_loader, num_epochs, loss_fn, optimizer, device):
    print("Training Started.")

    # Lists to store loss and accuracy for each epoch
    train_losses, train_accuracies = [], []
    val_losses, val_accuracies = [], []

    # Training loop
    for epoch in range(num_epochs):
        # Clear any cached GPU memory
        torch.cuda.empty_cache()

        # Run training and validation phases
        train_loss, train_accuracy = run_epoch(model, train_loader, loss_fn, optimizer, device, train=True)
        val_loss, val_accuracy = run_epoch(model, val_loader, loss_fn, optimizer, device, train=False)

        # Store metrics
        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)

        # Print epoch metrics
        print(f"Epoch [{epoch + 1}/{num_epochs}], "
              f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy * 100:.2f}%, "
              f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy * 100:.2f}%")

        # Save model weights every few epochs
        if (epoch + 1) % 5 == 0:
            torch.save(model.state_dict(), f'../data/asl_crop_v2_mobilenet_weights_epoch_{epoch + 1}.pth')
            print(f"Model weights saved for epoch {epoch + 1}.")

    print("Training finished.")

    # Plot training and validation loss and accuracy
    plot_performance_history(train_losses, val_losses, train_accuracies, val_accuracies)
