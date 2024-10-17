# utils/evaluation_tools.py

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report


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
