# train/train_model.py

from model import *


def train_model(model, dataloader, num_epochs, loss_fn, optimizer, device):
    print("Training Started.")

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
            torch.save(model.state_dict(), f'../data/asl_crop_v2_mobilenet_weights_epoch_{epoch + 1}.pth')
            print(f"Model weights saved for epoch {epoch + 1}.")

    print("Training finished.")
