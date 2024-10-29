# utils/model_checkpoint.py

from model import *


def save_checkpoint(model, optimizer, epoch, loss, filepath):
    # Save model, optimizer states, current epoch, and other training parameters
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
        'loss': loss,
    }
    torch.save(checkpoint, filepath)


def load_checkpoint(filepath, model, optimizer, device):
    # Load the checkpoint to resume training
    checkpoint = torch.load(filepath)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    return model, optimizer, epoch, loss


def load_model(filepath, device):
    num_classes = len(string.ascii_uppercase)
    model = CustomMobileNetV2(num_classes)
    state_dict = torch.load(filepath, weights_only=True, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    return model
