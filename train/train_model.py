# train/train_model.py


from utils import *


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
        avg_loss = running_loss / len(dataloader)
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(dataloader):.4f}")

        # Save a checkpoint every 5 epochs
        if (epoch + 1) % 5 == 0:
            checkpoint_filepath = f'checkpoint_epoch_{epoch + 1}.pth'
            save_checkpoint(model, optimizer, epoch, avg_loss, checkpoint_filepath)
            print(f"Checkpoint saved at epoch {epoch + 1}.")

    print("Training finished.")
