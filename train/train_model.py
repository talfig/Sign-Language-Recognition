# train/train_model.py

import matplotlib.pyplot as plt
from model import *


def train_model(model, dataloader, num_epochs, loss_fn, optimizer, device):
    print("Training Started.")

    # Lists to store loss and accuracy for each epoch
    epoch_losses = []
    epoch_accuracies = []

    # Training loop
    for epoch in range(num_epochs):
        # Clear any cached GPU memory
        torch.cuda.empty_cache()

        running_loss = 0.0
        correct_preds = 0
        total_preds = 0
        model.train()  # Set the model to training mode

        for images, labels in dataloader:
            # Move images and labels to the appropriate device (GPU or CPU)
            images, labels = images.to(device), labels.to(device)

            # Zero the gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(images)

            # Calculate loss and backpropagate
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            # Calculate accuracy using torch.eq
            _, preds = torch.max(outputs, 1)
            correct_preds += torch.eq(preds, labels).float().sum()  # Convert bools to float, then sum
            total_preds += labels.size(0)

        # Calculate epoch loss and accuracy
        epoch_loss = running_loss / len(dataloader)
        epoch_accuracy = (correct_preds / total_preds).item()
        epoch_losses.append(epoch_loss)
        epoch_accuracies.append(epoch_accuracy)

        # Print loss and accuracy for this epoch
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy * 100:.2f}%")

        # Save model weights every few epochs (e.g., every 5 epochs)
        if (epoch + 1) % 5 == 0:
            torch.save(model.state_dict(), f'../data/asl_crop_v2_mobilenet_weights_epoch_test{epoch + 1}.pth')
            print(f"Model weights saved for epoch {epoch + 1}.")

    print("Training finished.")

    # Plot Loss and Accuracy over Epochs
    plt.figure(figsize=(12, 5))

    # Loss subplot
    plt.subplot(1, 2, 1)
    plt.plot(epoch_losses, label='Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss over Epochs')
    plt.legend()

    # Accuracy subplot
    plt.subplot(1, 2, 2)
    plt.plot(epoch_accuracies, label='Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Accuracy over Epochs')
    plt.legend()

    plt.tight_layout()
    plt.show()
