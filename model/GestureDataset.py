# model/GestureDataset.py


import torch
import numpy as np
from torch.utils.data import Dataset
from PIL import Image


class GestureDataset(Dataset):
    def __init__(self, images, labels, label_map=None, transform=None):
        """
        Args:
            images: The dataset of images.
            labels: The corresponding labels for each image.
            label_map: A dictionary mapping original labels to new numerical labels.
            transform: Any transformations to apply to the images.
        """
        self.images = images  # Assuming images is a list/array of individual images
        self.labels = labels  # Corresponding labels for each image
        self.transform = transform

        # If label_map is provided, use it; otherwise, create a default one
        if label_map is None:
            self.label_map = {label: idx for idx, label in enumerate(set(labels))}
        else:
            self.label_map = label_map

        # Convert the labels to numerical format based on the label_map
        self.numerical_labels = np.array([self.label_map[label] for label in labels])

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image = self.images[idx]  # Access the image at the given index
        label = self.numerical_labels[idx]  # Access the corresponding numerical label

        # Convert the image to a PIL Image and apply any transforms
        if self.transform:
            image = self.transform(Image.fromarray(image))  # Apply transformations

        label = torch.tensor(label, dtype=torch.long)  # Change dtype as needed

        return image, label
