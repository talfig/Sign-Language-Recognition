# compressor/data_compressor.py

import os
import cv2
import numpy as np


# Resize and save images in compressed .npz format
def compress_to_npz(dataset_dir, output_file, image_size=(200, 200)):
    images = []
    labels = []

    # Loop through each class directory
    for label in os.listdir(dataset_dir):
        class_dir = os.path.join(dataset_dir, label)

        # Ensure the directory is a valid folder
        if os.path.isdir(class_dir):
            # Debug: print the current label and its directory
            print(f"Processing label: {label}, directory: {class_dir}")

            for filename in os.listdir(class_dir):
                filepath = os.path.join(class_dir, filename)

                # Read the image
                image = cv2.imread(filepath)

                # Ensure the image was read successfully
                if image is not None:
                    # Resize the image to the specified size
                    image_resized = cv2.resize(image, image_size)
                    images.append(image_resized)
                    labels.append(label)  # Append the label (class name)

    # Convert to numpy arrays and save
    images_np = np.array(images)
    labels_np = np.array(labels)

    # Save both images and labels in a structured array
    np.savez_compressed(output_file, images=images_np, labels=labels_np)


def decompress_npz(npz_file):
    # Load the compressed dataset
    data = np.load(npz_file)
    images = data['images']
    labels = data['labels']

    return images, labels


if __name__ == "__main__":
    compress_to_npz('../ASL-crop', '../data/compressed_asl_crop.npz')
