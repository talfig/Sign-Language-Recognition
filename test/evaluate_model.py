# test/evaluate_model.py

from sklearn.metrics import accuracy_score
from utils import *


def evaluate_model(model, dataloader, device):
    model.eval()  # Set the model to evaluation mode

    # Evaluate the model
    all_preds = []
    all_labels = []

    with torch.no_grad():  # Disable gradient computation during evaluation
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)

            # Forward pass (get predictions)
            outputs = model(images)

            # Get the predicted class (index of the max logit)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())  # Append predictions
            all_labels.extend(labels.cpu().numpy())  # Append true labels

    # Convert indexes to real class labels
    all_preds_labels = [LabelMapper.index_to_label(pred) for pred in all_preds]
    all_labels_labels = [LabelMapper.index_to_label(label) for label in all_labels]

    # Calculate accuracy
    accuracy = accuracy_score(all_labels_labels, all_preds_labels)
    print(f'Accuracy of the model on the test set: {accuracy * 100:.2f}%')

    # Confusion Matrix
    print_confusion_matrix(all_labels_labels, all_preds_labels)
