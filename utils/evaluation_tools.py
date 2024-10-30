# utils/evaluation_tools.py

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report


def plot_performance_history(train_losses, val_losses, train_accuracies, val_accuracies):
    """Plots the training and validation loss and accuracy over epochs with different colors."""
    plt.figure(figsize=(12, 5))

    # Loss subplot
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Training Loss', color='blue', linestyle='-')
    plt.plot(val_losses, label='Validation Loss', color='orange', linestyle='--')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss over Epochs')
    plt.legend()

    # Accuracy subplot
    plt.subplot(1, 2, 2)
    plt.plot(train_accuracies, label='Training Accuracy', color='blue', linestyle='-')
    plt.plot(val_accuracies, label='Validation Accuracy', color='orange', linestyle='--')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy over Epochs')
    plt.legend()

    plt.tight_layout()
    plt.show()


def print_confusion_matrix(y_true, y_pred, report=True):
    # Get sorted unique labels
    labels = sorted(list(set(y_true)))

    # Create confusion matrix
    cmx_data = confusion_matrix(y_true, y_pred, labels=labels)

    # Convert to DataFrame for better readability
    df_cmx = pd.DataFrame(cmx_data, index=labels, columns=labels)

    # Plot the confusion matrix
    fig, ax = plt.subplots(figsize=(10, 7))
    sns.heatmap(df_cmx, annot=True, fmt='g', cmap='Blues', square=False)

    # Ensure y-axis is displayed correctly
    ax.set_ylim(len(labels), 0)

    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.show()

    # Optional: print classification report
    if report:
        print('Classification Report')
        print(classification_report(y_true, y_pred))
