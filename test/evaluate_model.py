# test/evaluate_model.py


from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

from model import *
from compressor import *


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

    # Calculate accuracy
    accuracy = accuracy_score(all_labels, all_preds)
    print(f'Accuracy of the model on the test set: {accuracy * 100:.2f}%')

    # Confusion Matrix
    conf_matrix = confusion_matrix(all_labels, all_preds)

    # Plotting the confusion matrix
    plt.figure(figsize=(10, 7))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues",
                xticklabels=np.unique(all_labels),
                yticklabels=np.unique(all_labels))
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.show()
