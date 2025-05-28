# compressor/data_augmentation.py

import cv2
import numpy as np
import random

def augment_image(image, num_augmentations=2):
    """
    Apply a series of random augmentations (flip, zoom, shift) to the input image.

    This function is designed to improve model generalization by introducing
    variability into the dataset through random transformations.

    Args:
        image (np.ndarray): The input image as a NumPy array (H, W, C).
        num_augmentations (int): Number of different augmentation types to apply. Default is 2.

    Returns:
        np.ndarray: The augmented image.
    
    Augmentation types:
        - 'flip': Horizontally flips the image.
        - 'zoom': Randomly zooms in or out and crops/pads to original size.
        - 'shift': Translates the image in x and y directions.
    """
    augmentations = ['flip', 'zoom', 'shift']
    
    # Randomly select augmentation types to apply
    selected_augmentations = random.sample(augmentations, num_augmentations)

    for aug_type in selected_augmentations:
        if aug_type == 'flip':
            # Flip the image horizontally
            image = cv2.flip(image, 1)

        elif aug_type == 'zoom':
            # Apply a random zoom in or out
            zoom_factor = random.uniform(0.7, 1.3)
            h, w, _ = image.shape
            new_h, new_w = int(h * zoom_factor), int(w * zoom_factor)
            resized_image = cv2.resize(image, (new_w, new_h))

            if zoom_factor > 1.0:
                # Zoom in: crop the center
                x_start = (new_w - w) // 2
                y_start = (new_h - h) // 2
                image = resized_image[y_start:y_start + h, x_start:x_start + w]
            else:
                # Zoom out: pad the resized image into a canvas of original size
                canvas = np.zeros_like(image)
                x_offset = (w - new_w) // 2
                y_offset = (h - new_h) // 2
                canvas[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized_image
                image = canvas

        elif aug_type == 'shift':
            # Apply a random shift in both directions
            max_shift = 20
            dx, dy = random.randint(-max_shift, max_shift), random.randint(-max_shift, max_shift)
            matrix = np.float32([[1, 0, dx], [0, 1, dy]])
            image = cv2.warpAffine(image, matrix, (image.shape[1], image.shape[0]))

    return image
