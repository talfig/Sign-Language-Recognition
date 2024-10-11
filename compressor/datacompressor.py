# mypackage/datacompressor.py


import os
import cv2
import numpy as np


# Resize and save images in compressed .npz format
def compress2npz(dataset_dir, output_file):
    images = []
    labels = []

    # Loop through each class directory
    for label in os.listdir(dataset_dir):
        class_dir = os.path.join(dataset_dir, label)

        # Ensure the directory is a valid folder
        if os.path.isdir(class_dir):
            for filename in os.listdir(class_dir):
                filepath = os.path.join(class_dir, filename)

                # Read and resize image
                image = cv2.imread(filepath)

                # Ensure the image was read successfully
                if image is not None:
                    images.append(image)
                    labels.append(label)  # Append the label (class name)

    # Convert to numpy arrays and save
    images_np = np.array(images)
    labels_np = np.array(labels)

    # Save both images and labels in a structured array
    np.savez_compressed(output_file, images=images_np, labels=labels_np)
