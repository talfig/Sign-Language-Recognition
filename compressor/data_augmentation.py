# compressor/data_augmentation.py

import cv2
import numpy as np
import random


# Augmentation functions
def augment_image(image):
    # Apply random augmentation techniques (e.g., flipping, rotation, noise, resize)
    aug_type = random.choice(['flip', 'rotate', 'noise', 'resize'])

    if aug_type == 'flip':
        return cv2.flip(image, 1)  # Horizontal flip
    elif aug_type == 'rotate':
        angle = random.choice([90, 180, 270])
        center = (image.shape[1] // 2, image.shape[0] // 2)
        matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        return cv2.warpAffine(image, matrix, (image.shape[1], image.shape[0]))
    elif aug_type == 'noise':
        noise = np.random.randint(0, 50, image.shape, dtype='uint8')
        return cv2.add(image, noise)
    elif aug_type == 'resize':
        # Resize the entire image (smaller or larger) but keep the original dimensions
        h, w, _ = image.shape
        scale_factor = random.uniform(0.7, 1.3)  # Shrink to 70% or enlarge to 130%

        # Resize the image with the scale factor
        new_size = (int(w * scale_factor), int(h * scale_factor))
        resized_image = cv2.resize(image, new_size)

        # Create an empty canvas with the original size
        canvas = np.zeros_like(image)

        # If the image is smaller, center it in the canvas
        if scale_factor < 1.0:
            x_offset = (w - new_size[0]) // 2
            y_offset = (h - new_size[1]) // 2
            canvas[y_offset:y_offset + new_size[1], x_offset:x_offset + new_size[0]] = resized_image

        # If the image is larger, crop the center to fit the original size
        else:
            x_start = (new_size[0] - w) // 2
            y_start = (new_size[1] - h) // 2
            canvas = resized_image[y_start:y_start + h, x_start:x_start + w]

        return canvas

    return image
