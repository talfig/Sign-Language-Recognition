# compressor/data_compression.py

import os
from sklearn.utils import shuffle
from compressor.data_augmentation import *


# Function to get the number of images per class and the maximum count
def get_class_counts(dataset_dir):
    class_counts = {}

    # Loop through each class directory
    for label in os.listdir(dataset_dir):
        class_dir = os.path.join(dataset_dir, label)
        if os.path.isdir(class_dir):
            class_counts[label] = len(os.listdir(class_dir))  # Count images per class

    max_count = max(class_counts.values())  # Maximum number of images in any class
    return class_counts, max_count


# Resize, augment, and save images in compressed .npz format
def balance_compress_npz(dataset_dir, output_file, image_size=(224, 224), base_aug=1000):
    images = []
    labels = []

    # Get class counts and the maximum count
    class_counts, max_count = get_class_counts(dataset_dir)

    # Augment underrepresented classes
    for label in os.listdir(dataset_dir):
        class_dir = os.path.join(dataset_dir, label)
        if os.path.isdir(class_dir):
            label_images = []
            for filename in os.listdir(class_dir):
                filepath = os.path.join(class_dir, filename)
                image = cv2.imread(filepath)
                if image is not None:
                    image_resized = cv2.resize(image, image_size)
                    label_images.append(image_resized)

            # Augment images for underrepresented classes
            num_images_needed = max_count - len(label_images) + base_aug
            if num_images_needed > 0:
                augmented_images = []
                for i in range(num_images_needed):
                    image_to_augment = random.choice(label_images)
                    augmented_images.append(augment_image(image_to_augment))

                label_images.extend(augmented_images)

            images.extend(label_images)
            labels.extend([label] * len(label_images))
            
            # Debug: print after finishing processing a class
            print(f"Finished processing label: {label} - Total images: {len(label_images)}")

    # Shuffle the dataset
    images, labels = shuffle(np.array(images), np.array(labels))

    # Save to .npz
    np.savez_compressed(output_file, images=images, labels=labels)

    # Debug: print after saving the npz file
    print(f"Compressed and saved data to {output_file}")


def decompress_npz(npz_file):
    # Load the compressed dataset
    data = np.load(npz_file)
    images = data['images']
    labels = data['labels']

    return images, labels


if __name__ == "__main__":
    balance_compress_npz('../ASL-crop', '../data/compressed_asl_crop_v4.npz')
