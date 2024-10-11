# mypackage/datacompressor.py

import os
import cv2
import numpy as np


# Resize and save images in compressed .npz format
def compress2npz(dataset_dir, output_file, size=(200, 200)):
    images = []

    for filename in os.listdir(dataset_dir):
        filepath = os.path.join(dataset_dir, filename)

        # Read and resize image
        image = cv2.imread(filepath)
        resized_image = cv2.resize(image, size)

        # Add to list of images
        images.append(resized_image)

    # Convert to numpy array and save
    images_np = np.array(images)
    np.savez_compressed(output_file, images_np)
