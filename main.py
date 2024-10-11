import numpy as np
import matplotlib.pyplot as plt


def load_and_display_images(npz_file, num_images=5):
    # Load the compressed dataset
    data = np.load(npz_file)
    images = data['images']
    labels = data['labels']

    # Check if the number of images is less than the requested number
    num_images = min(num_images, images.shape[0])

    # Set up the plot
    plt.figure(figsize=(15, 10))

    # Display images
    for i in range(num_images):
        plt.subplot(1, num_images, i + 1)  # Create a subplot for each image
        plt.imshow(images[i])  # Display the image
        plt.axis('off')  # Hide axis
        plt.title(f'Label: {labels[i]}')  # Show label

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Example usage
    load_and_display_images('data/compressed_asl.npz', num_images=5)
