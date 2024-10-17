# model/gesture_dataset.py

import torch
import numpy as np
from torch.utils.data import Dataset
from utils.label_mapper import *


class GestureDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        """
        Initializes the dataset with images and labels.

        Args:
            images (list or array): The dataset of images.
            labels (list): Corresponding labels for each image ('0-9' and 'A-Z').
            transform (callable, optional): Transform to apply to images.
        """
        self.images = images  # List/array of images
        self.labels = labels  # Corresponding labels
        self.transform = transform

        # Instantiate LabelMapper
        self.label_mapper = LabelMapper()

        # Convert labels to numerical format
        self.numerical_labels = np.array([self.label_mapper.label_to_index(label) for label in labels])

    def __len__(self):
        """Returns the number of samples in the dataset."""
        return len(self.labels)

    def __getitem__(self, idx):
        """
        Retrieves an image and its corresponding label.

        Args:
            idx (int): Index of the sample.

        Returns:
            tuple: (image, label) where image is a PIL Image and label is a tensor.
        """
        image = self.images[idx]  # Get image at index
        label = self.numerical_labels[idx]  # Get corresponding label

        # Apply transformations if provided
        if self.transform:
            image = self.transform(image)

        label = torch.tensor(label, dtype=torch.long)  # Convert label to tensor

        return image, label
