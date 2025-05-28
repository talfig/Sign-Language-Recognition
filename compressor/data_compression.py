# compressor/data_compression.py

import os
from sklearn.utils import shuffle
from compressor.data_augmentation import *


def get_class_counts(dataset_dir):
    """
    Count the number of images in each class within a dataset directory.

    Args:
        dataset_dir (str): Path to the dataset directory, where each subdirectory represents a class.

    Returns:
        tuple:
            - class_counts (dict): A dictionary mapping each class label to the number of images it contains.
            - max_count (int): The maximum number of images in any class.
    """
    class_counts = {}

    # Loop through each class directory and count the number of image files
    for label in os.listdir(dataset_dir):
        class_dir = os.path.join(dataset_dir, label)
        if os.path.isdir(class_dir):
            class_counts[label] = len(os.listdir(class_dir))

    max_count = max(class_counts.values()) if class_counts else 0
    return class_counts, max_count


def balance_compress_npz(dataset_dir, output_file, image_size=(224, 224), base_aug=500):
    """
    Balances the dataset by augmenting underrepresented classes and compresses the images and labels into a .npz file.

    Args:
        dataset_dir (str): Path to the dataset directory with subdirectories for each class.
        output_file (str): Path where the compressed .npz file will be saved.
        image_size (tuple): Target size (width, height) to which all images will be resized. Default is (224, 224).
        base_aug (int): Base number of augmented images to add per class to increase robustness. Default is 500.

    Returns:
        None
    """
    images = []
    labels = []

    # Get the number of images per class and the maximum class size
    class_counts, max_count = get_class_counts(dataset_dir)

    for label in os.listdir(dataset_dir):
        class_dir = os.path.join(dataset_dir, label)
        if os.path.isdir(class_dir):
            label_images = []

            # Read and resize images for the current class
            for filename in os.listdir(class_dir):
                filepath = os.path.join(class_dir, filename)
                image = cv2.imread(filepath)
                if image is not None:
                    image_resized = cv2.resize(image, image_size)
                    label_images.append(image_resized)

            # Determine how many augmented images are needed to match the largest class
            num_images_needed = max_count - len(label_images) + base_aug
            if num_images_needed > 0:
                augmented_images = []
                for i in range(num_images_needed):
                    image_to_augment = random.choice(label_images)
                    augmented_images.append(augment_image(image_to_augment))

                label_images.extend(augmented_images)

            images.extend(label_images)
            labels.extend([label] * len(label_images))

            print(f"Finished processing label: {label} - Total images: {len(label_images)}")

    images, labels = shuffle(np.array(images), np.array(labels))

    np.savez_compressed(output_file, images=images, labels=labels)
    print(f"Compressed and saved data to {output_file}")


def decompress_npz(npz_file):
    """
    Loads and decompresses a dataset from a .npz file.

    Args:
        npz_file (str): Path to the .npz file containing compressed image data and labels.

    Returns:
        tuple:
            - images (np.ndarray): Array of images.
            - labels (np.ndarray): Array of corresponding labels.
    """
    data = np.load(npz_file)
    images = data['images']
    labels = data['labels']

    return images, labels


# Example usage
if __name__ == "__main__":
    balance_compress_npz('../ASL-crop', '../data/compressed_asl_crop_v4.npz')
