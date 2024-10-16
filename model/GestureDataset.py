# model/GestureDataset.py


import torch
import numpy as np
from torch.utils.data import Dataset
from PIL import Image


class GestureDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images  # Assuming images is a list/array of individual images
        self.labels = labels  # Corresponding labels for each image
        self.transform = transform

        # Convert string labels to numerical labels if necessary
        self.label_to_index = {label: idx for idx, label in enumerate(set(labels))}
        self.numerical_labels = np.array([self.label_to_index[label] for label in labels])

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image = self.images[idx]  # Access the image at the given index

        # Access the corresponding numerical label
        label = self.numerical_labels[idx]

        # Convert the image to a PIL Image and apply any transforms
        if self.transform:
            image = self.transform(Image.fromarray(image))  # Apply transformations

        label = torch.tensor(label, dtype=torch.long)  # Change dtype as needed

        return image, label
